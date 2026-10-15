from __future__ import annotations

import pytest
import torch

from src.common.cuda_fft import NormalizedInversePlan, ReconstructionIFFT
from src.common.cuda_graph import CudaGraphScorer
from src.detection.depth_runtime import FrameTensorRTScore, retain_winning_crops, run_model_batches

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def model_info(mode, inputs=2):
    return {"model_backend": "tensorrt", "input_mode": mode, "mask_radius_diam_scale": 2.0,
            "mask_min_radius_px": 1.0, "mask_softness_px": 3.0, "trt_num_inputs": inputs}


@pytest.mark.parametrize("mode", ["raw", "raw_diam_scalar", "raw_mask", "raw_mask_diam_scalar", "raw_mask_diam"])
@pytest.mark.parametrize("inputs", [1, 2])
def test_conditioning_reuse_preserves_inputs_batches_and_frame_reset(mode, inputs):
    info = model_info(mode, inputs)
    calls = []

    def recorder(x, cond=None):
        calls.append((x.clone(), cond.clone() if cond is not None else None))
        return x[:, 0].mean((1, 2))

    with torch.inference_mode():
        # New frame changes diameters and particle order, requiring a new scorer.
        for diameters in (torch.tensor([25., 50., 80., 170., 300.]), torch.tensor([300., 25., 80., 50., 170.])):
            cached = FrameTensorRTScore(recorder, info, 3, 8.0)
            for count, dtype in ((5, torch.float32), (5, torch.float32), (2, torch.float32), (5, torch.float64)):
                crops = torch.rand((count, 9, 9), dtype=dtype)
                calls.clear()
                expected = run_model_batches(recorder, crops, diameters[:count], info, 3, False, False, 8.0)
                reference_calls = calls.copy()
                calls.clear()
                actual = cached(crops, diameters[:count])
                assert torch.equal(actual, expected)
                assert len(calls) == len(reference_calls)
                for (x, c), (rx, rc) in zip(calls, reference_calls, strict=True):
                    assert torch.equal(x, rx)
                    assert (c is None and rc is None) or torch.equal(c, rc)


def test_conditioning_backend_guard_and_grad_mode_do_not_cache_inputs():
    with pytest.raises(ValueError, match="TensorRT"):
        FrameTensorRTScore(None, {"model_backend": "torch"}, 3, 10.0)
    score = FrameTensorRTScore(lambda x, d: x.mean((1, 2, 3)), model_info("raw_diam_scalar"), 3, 10.0)
    crops, diam = torch.rand((2, 5, 5)), torch.tensor([25., 80.])
    expected = run_model_batches(score.model, crops, diam, score.info, 3, False, False)
    assert torch.equal(score(crops, diam), expected)
    assert score.inputs is None


def test_winning_crops_keep_routed_argmax_ties_default_slice_and_independent_storage():
    # The two rows deliberately include a tie / no finite score. Scores here are
    # already routed to the primary or fallback model, as in the frame loop.
    blocks = [torch.arange(24.).reshape(3, 2, 2, 2), torch.arange(24., 40.).reshape(2, 2, 2, 2)]
    scores = [torch.tensor([[1., -torch.inf], [2., -torch.inf], [2., -torch.inf]]),
              torch.tensor([[2., -torch.inf], [3., -torch.inf]])]
    best_scores = torch.full((2,), -torch.inf)
    selected = None
    for crops, block_scores in zip(blocks, scores, strict=True):
        values, arg = block_scores.max(0)
        improve = values > best_scores
        selected = retain_winning_crops(crops, arg, improve, selected)
        best_scores = torch.where(improve, values, best_scores)
    assert torch.equal(selected, torch.stack((blocks[1][1, 0], blocks[0][0, 1])))
    expected = selected.clone()
    for b in blocks:
        b.fill_(999)
    assert torch.equal(selected, expected)
    first = torch.rand(2, 2, 3, 3)
    values, arg = torch.full((2, 2), float("nan")).max(0)
    assert torch.equal(retain_winning_crops(first, arg, values > -torch.inf), first[0])


def test_cpu_fft_and_autograd_preserve_reference():
    x = torch.randn(9, 9, dtype=torch.complex128, requires_grad=True)
    with ReconstructionIFFT(9, torch.device("cpu"), enabled=True) as inverse:
        actual = inverse(x)
        assert not inverse.active
        assert torch.equal(actual, torch.fft.ifft2(x))
        actual.real.sum().backward()
        assert x.grad is not None


@requires_cuda
@pytest.mark.parametrize("side", [16, 63, 64, 1536])
def test_fused_ifft_repeated_fields_match_and_do_not_mutate_frequency(side):
    pytest.importorskip("nvmath")
    device = torch.device("cuda:0")
    with torch.inference_mode(), ReconstructionIFFT(side, device, enabled=True) as inverse:
        assert inverse.active, inverse.fallback_reason
        for _ in range(3):
            field = torch.randn((side, side), device=device, dtype=torch.complex64)
            original = field.clone()
            expected = torch.fft.ifft2(field)
            assert torch.equal(inverse(field), expected)
            assert torch.equal(field, original)


@requires_cuda
def test_fused_ifft_special_values_stream_ownership_and_cleanup():
    pytest.importorskip("nvmath")
    device = torch.device("cuda:0")
    with torch.inference_mode(), NormalizedInversePlan(16, device) as plan:
        for value in (1e-38, float("inf"), float("nan")):
            x = torch.zeros((16, 16), dtype=torch.complex64, device=device)
            x[0, 0] = value
            torch.testing.assert_close(plan.execute(x), torch.fft.ifft2(x), rtol=0, atol=0, equal_nan=True)
        with pytest.raises(ValueError, match="overlap"):
            plan.execute(plan.output)
        with pytest.raises(ValueError, match="contiguous"):
            plan.execute(torch.zeros(16, 16, dtype=torch.complex128, device=device))
        with torch.cuda.stream(torch.cuda.Stream(device=device)), pytest.raises(RuntimeError, match="stream"):
            plan.execute(x)
    plan.close()
    with pytest.raises(RuntimeError, match="closed"):
        plan.execute(x)


@requires_cuda
def test_fft_preflight_failure_closes_plan_and_falls_back(monkeypatch):
    import src.common.cuda_fft as module

    closed = []

    class WrongPlan:
        def __init__(self, side, device):
            pass

        def execute(self, x):
            return torch.zeros_like(x)

        def close(self):
            closed.append(True)

    monkeypatch.setattr(module, "NormalizedInversePlan", WrongPlan)
    with pytest.warns(RuntimeWarning, match="preflight"), torch.inference_mode():
        with ReconstructionIFFT(16, torch.device("cuda:0"), enabled=True) as inverse:
            x = torch.randn(16, 16, dtype=torch.complex64, device="cuda:0")
            assert not inverse.active and inverse.plan is None
            assert torch.equal(inverse(x), torch.fft.ifft2(x))
    assert closed == [True]


@requires_cuda
def test_unavailable_fft_library_falls_back_before_any_frame(monkeypatch):
    import src.common.cuda_fft as module

    def unavailable(*_):
        raise ImportError("optional FFT binding unavailable")

    monkeypatch.setattr(module, "NormalizedInversePlan", unavailable)
    with pytest.warns(RuntimeWarning, match="optional FFT binding"), torch.inference_mode():
        with ReconstructionIFFT(16, torch.device("cuda:0"), enabled=True) as inverse:
            assert not inverse.active and inverse.plan is None
            assert inverse.fallback_reason.startswith("ImportError:")
            x = torch.randn(16, 16, dtype=torch.complex64, device="cuda:0")
            assert torch.equal(inverse(x), torch.fft.ifft2(x))


@requires_cuda
def test_fft_adapter_uses_torch_for_other_layouts_and_gradients_and_closes_on_exception():
    pytest.importorskip("nvmath")
    device = torch.device("cuda:0")
    with pytest.raises(RuntimeError, match="frame failed"):
        with ReconstructionIFFT(16, device, enabled=True) as inverse:
            assert inverse.active
            plan = inverse.plan
            with torch.inference_mode():
                for x in (torch.randn(16, 16, dtype=torch.complex128, device=device),
                          torch.randn(16, 16, dtype=torch.complex64, device=device).T,
                          torch.randn(8, 8, dtype=torch.complex64, device=device)):
                    assert torch.equal(inverse(x), torch.fft.ifft2(x))
            x = torch.randn(16, 16, dtype=torch.complex64, device=device, requires_grad=True)
            result = inverse(x)
            assert torch.equal(result, torch.fft.ifft2(x))
            result.real.square().sum().backward()
            assert x.grad is not None
            raise RuntimeError("frame failed")
    assert inverse.plan is None and plan.handle is None


@requires_cuda
def test_graph_replays_cached_inputs_with_changed_images_and_partial_block():
    info = model_info("raw_diam_scalar")

    def model(x, cond):
        return x[:, 0].mean((1, 2)) + cond[:, 0]

    with torch.inference_mode():
        diam = torch.tensor([25., 50., 80., 150., 300.], device="cuda:0")
        cached = FrameTensorRTScore(model, info, 3, 10.0)
        scorer = CudaGraphScorer(cached, enabled=True)
        for count in (5, 5, 2, 5):
            crops = torch.rand((count, 9, 9), device="cuda:0")
            expected = run_model_batches(model, crops, diam[:count], info, 3, False, False)
            assert torch.equal(scorer(crops, diam[:count]), expected)
        assert scorer.captures == 1 and scorer.replays == 3
        assert scorer.fallback_reason is None
