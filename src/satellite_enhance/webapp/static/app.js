// ---- tiny state ----
const state = {
  file2d: null,
  sample2d: null,
  file3d: null,
  sample3d: null,
  frames3d: { before: [], after: [] },
};

// ---- tabs ----
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  });
});

function showLoading(text) {
  document.getElementById("loading-text").textContent = text || "Working...";
  document.getElementById("loading-overlay").classList.remove("hidden");
}
function hideLoading() {
  document.getElementById("loading-overlay").classList.add("hidden");
}
function showError(msg) {
  const toast = document.createElement("div");
  toast.className = "error-toast";
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4500);
}

// ==================================================================
// 2D tab
// ==================================================================
const dropzone2d = document.getElementById("dropzone-2d");
const fileInput2d = document.getElementById("file-2d");
const btnEnhance2d = document.getElementById("btn-enhance-2d");

dropzone2d.addEventListener("click", () => fileInput2d.click());
["dragover", "dragleave", "drop"].forEach((evt) =>
  dropzone2d.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone2d.classList.toggle("dragover", evt === "dragover");
  })
);
dropzone2d.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) selectFile2d(f);
});
fileInput2d.addEventListener("change", (e) => {
  if (e.target.files[0]) selectFile2d(e.target.files[0]);
});

function selectFile2d(file) {
  state.file2d = file;
  state.sample2d = null;
  document.querySelectorAll("#samples-2d .sample-thumb").forEach((el) => el.classList.remove("selected"));
  dropzone2d.querySelector("p").textContent = `Selected: ${file.name}`;
  btnEnhance2d.disabled = false;
}

async function loadSamples2d() {
  const res = await fetch("/api/samples/2d");
  const samples = await res.json();
  const grid = document.getElementById("samples-2d");
  grid.innerHTML = "";
  samples.forEach((s) => {
    const div = document.createElement("div");
    div.className = "sample-thumb";
    div.innerHTML = `<img src="${s.url}" loading="lazy" /><span class="label">${s.name.replace(/\.[a-z]+$/, "")}</span>`;
    div.addEventListener("click", () => {
      state.sample2d = s.name;
      state.file2d = null;
      dropzone2d.querySelector("p").textContent = "Drag & drop an image, or click to choose a file";
      document.querySelectorAll("#samples-2d .sample-thumb").forEach((el) => el.classList.remove("selected"));
      div.classList.add("selected");
      btnEnhance2d.disabled = false;
    });
    grid.appendChild(div);
  });
}

btnEnhance2d.addEventListener("click", async () => {
  const form = new FormData();
  if (state.file2d) form.append("file", state.file2d);
  else if (state.sample2d) form.append("sample", state.sample2d);
  else return;

  showLoading("Running 2D super-resolution...");
  try {
    const res = await fetch("/api/enhance/2d", { method: "POST", body: form });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();
    renderResult2d(data);
  } catch (err) {
    showError(`2D enhancement failed: ${err.message}`);
  } finally {
    hideLoading();
  }
});

function renderResult2d(data) {
  document.getElementById("result-2d-empty").classList.add("hidden");
  document.getElementById("result-2d").classList.remove("hidden");

  const bicubicImg = document.getElementById("img-bicubic");
  const enhancedImg = document.getElementById("img-enhanced");
  bicubicImg.src = data.bicubic_png;
  enhancedImg.src = data.enhanced_png;

  const compare = document.getElementById("compare-2d");
  const updateOverlayWidth = () => {
    compare.style.setProperty("--overlay-img-w", compare.clientWidth + "px");
  };
  enhancedImg.onload = updateOverlayWidth;
  window.addEventListener("resize", updateOverlayWidth);

  const slider = document.getElementById("compare-slider");
  const overlay = document.getElementById("compare-overlay");
  slider.oninput = () => (overlay.style.width = slider.value + "%");
  overlay.style.width = slider.value + "%";

  document.getElementById("stats-2d").innerHTML = `
    <span>Input: <b>${data.input_size[0]}×${data.input_size[1]}</b></span>
    <span>Output: <b>${data.output_size[0]}×${data.output_size[1]}</b></span>
    <span>Scale: <b>×${data.scale}</b></span>
    <span>Inference time: <b>${data.elapsed_sec}s</b></span>
  `;

  document.getElementById("download-2d").href = data.enhanced_png;
}

// ==================================================================
// 3D tab
// ==================================================================
const dropzone3d = document.getElementById("dropzone-3d");
const fileInput3d = document.getElementById("file-3d");
const btnEnhance3d = document.getElementById("btn-enhance-3d");
const zrangeInputs = document.getElementById("zrange-inputs");

dropzone3d.addEventListener("click", () => fileInput3d.click());
["dragover", "dragleave", "drop"].forEach((evt) =>
  dropzone3d.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone3d.classList.toggle("dragover", evt === "dragover");
  })
);
dropzone3d.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) selectFile3d(f);
});
fileInput3d.addEventListener("change", (e) => {
  if (e.target.files[0]) selectFile3d(e.target.files[0]);
});

function selectFile3d(file) {
  state.file3d = file;
  state.sample3d = null;
  document.querySelectorAll("#samples-3d .sample-thumb").forEach((el) => el.classList.remove("selected"));
  dropzone3d.querySelector("p").textContent = `Selected: ${file.name}`;
  zrangeInputs.classList.remove("hidden");
  btnEnhance3d.disabled = false;
}

async function loadSamples3d() {
  const res = await fetch("/api/samples/3d");
  const samples = await res.json();
  const grid = document.getElementById("samples-3d");
  grid.innerHTML = "";
  samples.forEach((s) => {
    const div = document.createElement("div");
    div.className = "sample-thumb";
    div.innerHTML = `<img src="/api/samples/3d/${s.name}/preview" loading="lazy" /><span class="label">${s.z_min}-${s.z_max}m</span>`;
    div.addEventListener("click", () => {
      state.sample3d = s.name;
      state.file3d = null;
      dropzone3d.querySelector("p").textContent = "Drag & drop a 16-bit grayscale DEM PNG/TIFF";
      zrangeInputs.classList.add("hidden");
      document.querySelectorAll("#samples-3d .sample-thumb").forEach((el) => el.classList.remove("selected"));
      div.classList.add("selected");
      btnEnhance3d.disabled = false;
    });
    grid.appendChild(div);
  });
}

btnEnhance3d.addEventListener("click", async () => {
  const form = new FormData();
  if (state.file3d) {
    form.append("file", state.file3d);
    form.append("z_min", document.getElementById("z-min").value);
    form.append("z_max", document.getElementById("z-max").value);
  } else if (state.sample3d) {
    form.append("sample", state.sample3d);
  } else return;

  showLoading("Running 3D DEM super-resolution + rendering views...");
  try {
    const res = await fetch("/api/enhance/3d", { method: "POST", body: form });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();
    renderResult3d(data);
  } catch (err) {
    showError(`3D enhancement failed: ${err.message}`);
  } finally {
    hideLoading();
  }
});

function renderResult3d(data) {
  document.getElementById("result-3d-empty").classList.add("hidden");
  document.getElementById("result-3d").classList.remove("hidden");

  state.frames3d.before = data.before_frames;
  state.frames3d.after = data.after_frames;

  const slider = document.getElementById("spin-slider");
  slider.max = data.before_frames.length - 1;
  slider.value = 0;
  updateSpinFrame(0);
  slider.oninput = () => updateSpinFrame(slider.value);

  document.getElementById("stats-3d").innerHTML = `
    <span>Input grid: <b>${data.input_shape[1]}×${data.input_shape[0]}</b></span>
    <span>Output grid: <b>${data.output_shape[1]}×${data.output_shape[0]}</b></span>
    <span>Scale: <b>×${data.scale}</b></span>
    <span>Elevation range: <b>${data.elevation_range_m[0].toFixed(0)}-${data.elevation_range_m[1].toFixed(0)}m</b></span>
    <span>Inference time: <b>${data.elapsed_sec}s</b></span>
  `;

  document.getElementById("download-3d").href = `/api/mesh/${data.mesh_id}.obj`;
}

function updateSpinFrame(idx) {
  document.getElementById("spin-before").src = state.frames3d.before[idx];
  document.getElementById("spin-after").src = state.frames3d.after[idx];
}

// ---- init ----
loadSamples2d();
loadSamples3d();
