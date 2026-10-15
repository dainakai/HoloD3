from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from holod3.acquisition import AcquisitionConfig
from holod3.config import repository_root
from holod3.reconstruction import (
    PropagationSetup,
    build_propagation_setup,
    load_wavefront,
    mean_pad_wavefront,
    minimum_intensity_projection,
)
from src.common.cuda_graph import CudaGraphScorer

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def legacy_minip(wave: torch.Tensor, setup: PropagationSetup, step: int) -> torch.Tensor:
    """Independent copy of the pre-0.2.1 recurrence and reduction order."""
    offset = (setup.padding_side - setup.image_size) // 2
    freq = torch.fft.fft2(mean_pad_wavefront(wave, setup.padding_side)) * setup.initial_transfer
    result = None
    previous = 0
    for index in range(0, setup.slice_count, max(1, int(step))):
        if index != previous:
            freq = freq * (setup.slice_transfer ** (index - previous))
        previous = index
        field = torch.fft.ifft2(freq)
        intensity = (field.real.square() + field.imag.square()).clamp_(0, 1)
        central = intensity[offset:offset + setup.image_size, offset:offset + setup.image_size]
        result = central if result is None else torch.minimum(result, central)
    assert result is not None
    return result


def synthetic_setup(device: str, image: int = 9, padding: int = 12, slices: int = 17) -> tuple:
    generator = torch.Generator().manual_seed(191)
    wave = torch.randn((image, image), dtype=torch.complex64, generator=generator).to(device)
    phase = torch.randn((padding, padding), generator=generator).to(device)
    setup = PropagationSetup(
        image, padding, slices, 10.0, 100.0, None, None,
        torch.exp(1j * phase), torch.exp(1j * phase * 0.013), None,
    )
    return wave, setup


@pytest.mark.parametrize("step", [0, 1, 2, 3, 30])
def test_cpu_minip_preserves_reference(step: int) -> None:
    wave, setup = synthetic_setup("cpu")
    actual = minimum_intensity_projection(wave, setup, slice_step=step)
    assert torch.equal(actual, legacy_minip(wave, setup, step))
    with pytest.raises(RuntimeError, match="CUDA MinIP requires"):
        minimum_intensity_projection(wave, setup, backend="cuda")
    with pytest.raises(ValueError, match="backend"):
        minimum_intensity_projection(wave, setup, backend="invalid")


def test_cpu_graph_scorer_preserves_gradients_and_disabled_execution() -> None:
    crops = torch.ones((2, 3, 3), requires_grad=True)
    diam = torch.tensor([2.0, 3.0])
    scorer = CudaGraphScorer(lambda x, d: x.sum((1, 2)) + d, enabled=True)
    scorer(crops, diam).sum().backward()
    assert torch.equal(crops.grad, torch.ones_like(crops))
    assert scorer.captures == 0


@requires_cuda
@pytest.mark.parametrize("step", [1, 2, 3, 30])
@pytest.mark.parametrize("image,padding,slices", [(9, 12, 17), (16, 16, 1), (31, 48, 9)])
def test_cuda_minip_matches_legacy_for_padding_and_strides(step: int, image: int, padding: int, slices: int) -> None:
    wave, setup = synthetic_setup("cuda:0", image, padding, slices)
    with torch.inference_mode():
        expected = legacy_minip(wave, setup, step)
        actual = minimum_intensity_projection(wave, setup, slice_step=step, backend="cuda")
    assert torch.equal(actual, expected)


@requires_cuda
def test_cuda_minip_preserves_clipping_nan_subnormals_and_current_stream() -> None:
    from src.common.cuda_minip_kernels import CudaMinIPUpdater

    device = torch.device(f"cuda:{torch.cuda.device_count() - 1}")
    updater = CudaMinIPUpdater.build(device.index)
    with torch.cuda.device(device), torch.cuda.stream(torch.cuda.Stream(device=device)):
        field = torch.zeros((6, 6), dtype=torch.complex64, device=device)
        field[1:5, 1:5] = torch.tensor([
            [0, 0.3 + 0.4j, 2, float("nan")],
            [float("inf"), 1e-20, 0.75 + 0.5j, -0.0],
            [1e-19j, -2j, 1j, 1e-10 + 1e-10j],
            [-0.1, 0.8j, -0.9j, 0.1 + 0.2j],
        ], device=device)
        actual = torch.ones((4, 4), device=device)
        expected = torch.ones_like(actual)
        for _ in range(2):
            intensity = (field.real.square() + field.imag.square()).clamp_(0, 1)
            expected = torch.minimum(expected, intensity[1:5, 1:5])
            updater.update(field, actual, 1)
            field.mul_(0.5)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        with pytest.raises(ValueError, match="contiguous"):
            updater.update(field.T, actual, 1)


@requires_cuda
def test_cuda_minip_auto_falls_back_without_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    wave, setup = synthetic_setup("cuda:0")
    monkeypatch.setattr("holod3.reconstruction._cuda_minip_updater", lambda _: None)
    assert torch.equal(minimum_intensity_projection(wave, setup), legacy_minip(wave, setup, 1))
    with pytest.raises(RuntimeError, match="CUDA MinIP requires"):
        minimum_intensity_projection(wave, setup, backend="cuda")


@requires_cuda
def test_cuda_minip_preserves_autograd_and_complex128_reference() -> None:
    wave, setup = synthetic_setup("cuda:0")
    wave = wave.to(torch.complex128)
    assert torch.equal(minimum_intensity_projection(wave, setup), legacy_minip(wave, setup, 1))
    wave = wave.to(torch.complex64)
    setup.initial_transfer.requires_grad_(True)
    actual = minimum_intensity_projection(wave, setup)
    expected = legacy_minip(wave, setup, 1)
    assert torch.equal(actual, expected)
    left = torch.autograd.grad(actual.sum(), setup.initial_transfer, retain_graph=True)[0]
    right = torch.autograd.grad(expected.sum(), setup.initial_transfer)[0]
    assert torch.equal(left, right)


@requires_cuda
def test_graph_replay_updates_inputs_and_bounds_shape_memory() -> None:
    def score(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return ((x - 0.5) / 0.25).sum((1, 2)) + d

    scorer = CudaGraphScorer(score, enabled=True)
    with torch.inference_mode():
        crops = torch.arange(32 * 16 * 16, device="cuda:0", dtype=torch.float32).view(32, 16, 16) / 1000
        diam = torch.arange(32, device="cuda:0", dtype=torch.float32)
        assert torch.equal(scorer(crops, diam), score(crops, diam))
        for scale in (0.1, 0.7, 1.3):
            crops.mul_(scale)
            diam.add_(1)
            output = scorer(crops, diam)
            assert torch.equal(output, score(crops, diam))
            output.fill_(float("nan"))  # callers may overwrite fallback scores
        assert torch.equal(scorer(crops[:5], diam[:5]), score(crops[:5], diam[:5]))
        assert torch.equal(scorer(crops, diam), score(crops, diam))
    assert scorer.captures == 1 and scorer.replays == 5
    assert scorer.fallback_reason is None


@requires_cuda
@pytest.mark.parametrize("failure", ["capture", "parity"])
@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty:UserWarning")
def test_graph_failure_keeps_same_eager_function(failure: str) -> None:
    def score(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        if torch.cuda.is_current_stream_capturing():
            if failure == "capture":
                raise RuntimeError("fixture cannot be captured")
            return x.sum((1, 2)) + d + 1
        return x.sum((1, 2)) + d

    scorer = CudaGraphScorer(score, enabled=True)
    with torch.inference_mode():
        crops, diam = torch.ones((8, 4, 4), device="cuda:0"), torch.ones(8, device="cuda:0")
        with pytest.warns(RuntimeWarning, match="same model eagerly"):
            assert torch.equal(scorer(crops, diam), score(crops, diam))
        assert torch.equal(scorer(crops * 2, diam), score(crops * 2, diam))
    assert scorer.captures == 0 and scorer.graph is None and not scorer.enabled


@pytest.mark.integration
@pytest.mark.slow
@requires_cuda
def test_all_demo_minips_match_legacy_on_both_modes() -> None:
    original = AcquisitionConfig.load(repository_root() / "data/demo/experimental/acquisition.yaml")
    for mode in ("dual_phase_retrieval", "single_gabor"):
        config = replace(original, mode=mode)
        setup = build_propagation_setup(config, torch.device("cuda:0"))
        with torch.inference_mode():
            for record in config.frame_records():
                wave = load_wavefront(config, record, setup, torch.device("cuda:0"))
                for step in (1, 2, 7):
                    expected = legacy_minip(wave, setup, step)
                    actual = minimum_intensity_projection(wave, setup, slice_step=step, backend="cuda")
                    assert torch.equal(expected, actual), (mode, record.stem, step)


@pytest.mark.integration
@requires_cuda
@pytest.mark.parametrize("checkpoint", ["depth-primary.pt", "depth-fallback.pt"])
def test_graph_packaged_depth_models_match_on_changing_inputs(checkpoint: str) -> None:
    from src.detection.depth_runtime import load_model, run_model_batches

    model, info = load_model(repository_root() / "models/production" / checkpoint, torch.device("cuda:0"), False, False)
    def score(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return run_model_batches(model, x, d, info, 256, False, False)

    scorer = CudaGraphScorer(score, enabled=True)
    with torch.inference_mode():
        for seed in (9, 31, 88):
            generator = torch.Generator().manual_seed(seed)
            crops = torch.rand((257, 64, 64), generator=generator).cuda()
            diam = (25 + torch.rand(257, generator=generator) * 475).cuda()
            assert torch.equal(scorer(crops, diam), score(crops, diam))
    assert scorer.captures == 1 and scorer.replays == 3
