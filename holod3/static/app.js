const detectForm = document.querySelector("#detect-form");
const detectStatus = document.querySelector("#detect-status");
const detectResult = document.querySelector("#detect-result");
const annotatedImage = document.querySelector("#annotated-image");
const detectionJson = document.querySelector("#detection-json");

async function responsePayload(response) {
  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch (_error) {
    return { error: text.trim() || `HTTP ${response.status}` };
  }
}

detectForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  detectStatus.textContent = "Running the production detector…";
  detectResult.classList.add("hidden");
  const form = new FormData();
  const file = document.querySelector("#image-file").files[0];
  if (file) form.append("image", file);
  else form.append("image_path", document.querySelector("#image-path").value);
  try {
    const response = await fetch("/api/detect", { method: "POST", body: form });
    const payload = await responsePayload(response);
    if (!response.ok) throw new Error(payload.error || "Detection failed.");
    detectStatus.textContent = `Found ${payload.count} particle detections.`;
    annotatedImage.src = `data:image/png;base64,${payload.annotated_png_base64}`;
    detectionJson.textContent = JSON.stringify({ count: payload.count, detections: payload.detections }, null, 2);
    detectResult.classList.remove("hidden");
  } catch (error) {
    detectStatus.textContent = error.message;
  }
});

const pipelineForm = document.querySelector("#pipeline-form");
const pipelineStatus = document.querySelector("#pipeline-status");
const pipelineResult = document.querySelector("#pipeline-result");
const pipelineLinks = document.querySelector("#pipeline-links");

async function pollJob(jobId) {
  const response = await fetch(`/api/jobs/${jobId}`);
  const state = await responsePayload(response);
  if (!response.ok) throw new Error(state.error || "Could not read pipeline status.");
  pipelineStatus.textContent = `${state.status}: ${state.message || "waiting"}`;
  pipelineResult.textContent = JSON.stringify(state, null, 2);
  pipelineResult.classList.remove("hidden");
  if (state.status === "complete" && state.result && state.result.downloads) {
    pipelineLinks.replaceChildren();
    const labels = {
      particles_csv: "Download particle CSV",
      summary_json: "Download run summary",
      visualization_html: "Open animated 3D scatter",
      log: "Download run log",
    };
    for (const [kind, url] of Object.entries(state.result.downloads)) {
      const link = document.createElement("a");
      link.href = url;
      link.textContent = labels[kind] || kind;
      if (kind === "visualization_html") link.target = "_blank";
      pipelineLinks.append(link);
    }
    pipelineLinks.classList.remove("hidden");
  }
  if (state.status === "queued" || state.status === "running") {
    window.setTimeout(() => pollJob(jobId), 1500);
  }
}

pipelineForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  pipelineStatus.textContent = "Creating pipeline job…";
  pipelineResult.classList.add("hidden");
  pipelineLinks.classList.add("hidden");
  const form = new FormData(pipelineForm);
  for (const name of ["bbox_fallback", "depth_router", "diameter_fallback"]) {
    form.set(name, pipelineForm.elements[name].checked ? "true" : "false");
  }
  try {
    const response = await fetch("/api/pipeline", { method: "POST", body: form });
    const payload = await responsePayload(response);
    if (!response.ok) throw new Error(payload.error || "Could not create pipeline job.");
    await pollJob(payload.job_id);
  } catch (error) {
    pipelineStatus.textContent = error.message;
  }
});
