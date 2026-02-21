const form = document.getElementById("predict-form");
const input = document.getElementById("video-input");
const dropZone = document.getElementById("drop-zone");
const submitBtn = document.getElementById("submit-btn");
const fileName = document.getElementById("file-name");
const preview = document.getElementById("video-preview");
const statusEl = document.getElementById("status");
const moveEl = document.getElementById("predicted-move");
const confidenceEl = document.getElementById("predicted-confidence");
const predictedModelEl = document.getElementById("predicted-model");
const framesEl = document.getElementById("detected-frames");
const selectedSegmentEl = document.getElementById("selected-segment");
const bestWindowEl = document.getElementById("best-window");
const windowsTestedEl = document.getElementById("windows-tested");
const probabilitiesEl = document.getElementById("probabilities");
const chipsEl = document.getElementById("move-chips");
const startTimeInput = document.getElementById("start-time");
const endTimeInput = document.getElementById("end-time");
const setStartBtn = document.getElementById("set-start");
const setEndBtn = document.getElementById("set-end");
const modelSelect = document.getElementById("model-select");
const refreshBenchmarkBtn = document.getElementById("refresh-benchmark");
const benchmarkStatusEl = document.getElementById("benchmark-status");
const benchmarkBestEl = document.getElementById("benchmark-best");
const benchmarkTableEl = document.getElementById("benchmark-table");
const liveVideo = document.getElementById("live-video");
const startLiveBtn = document.getElementById("start-live");
const stopLiveBtn = document.getElementById("stop-live");
const liveStatusEl = document.getElementById("live-status");
const liveMoveEl = document.getElementById("live-move");
const liveConfidenceEl = document.getElementById("live-confidence");
const menuToggleBtn = document.getElementById("menu-toggle");
const drawerCloseBtn = document.getElementById("drawer-close");
const drawerBackdrop = document.getElementById("drawer-backdrop");
const drawer = document.getElementById("menu-drawer");
const topLoginBtn = document.getElementById("top-login-btn");
const currentViewEl = document.getElementById("current-view");
const navLinks = document.querySelectorAll("[data-view-target]");
const viewSections = document.querySelectorAll(".view-section");
const loginForm = document.getElementById("login-form");
const loginStatusEl = document.getElementById("login-status");
const loginEmailInput = document.getElementById("login-email");
const flexForm = document.getElementById("flex-form");
const flexVideoInput = document.getElementById("flex-video-input");
const flexDropZone = document.getElementById("flex-drop-zone");
const flexFileNameEl = document.getElementById("flex-file-name");
const flexAnalyzeGoalBtn = document.getElementById("flex-analyze-goal-btn");
const flexGoalSelect = document.getElementById("flex-goal-select");
const flexAthleteIdInput = document.getElementById("flex-athlete-id");
const flexKnownScoresInput = document.getElementById("flex-known-scores");
const flexPreview = document.getElementById("flex-video-preview");
const flexStatusEl = document.getElementById("flex-status");
const flexPredictedMoveEl = document.getElementById("flex-predicted-move");
const flexQualityEl = document.getElementById("flex-quality");
const flexPoornessEl = document.getElementById("flex-poorness");
const flexReadinessEl = document.getElementById("flex-readiness");
const flexSessionIdEl = document.getElementById("flex-session-id");
const flexJsonEl = document.getElementById("flex-json");

const trainingModal = document.getElementById("training-modal");
const trainingPreview = document.getElementById("training-preview");
const trainingStartInput = document.getElementById("training-start");
const trainingEndInput = document.getElementById("training-end");
const trainingPredictedEl = document.getElementById("training-predicted");
const trainingConfidenceEl = document.getElementById("training-confidence");
const actualMoveSelect = document.getElementById("actual-move");
const modelCorrectInput = document.getElementById("model-correct");
const cancelTrainingBtn = document.getElementById("cancel-training");
const confirmTrainingBtn = document.getElementById("confirm-training");

let availableMoves = [];
let availableModels = [];
let lastPrediction = null;
let lastFile = null;
let liveStream = null;
let liveInterval = null;
let liveBusy = false;
let flexLastFile = null;

const VIEW_LABELS = {
  dashboard: "Dashboard",
  live: "Live Detection",
  "model-lab": "Model Lab",
  flexibility: "FlexibilityAI",
  about: "About CapoAI",
  login: "Login",
  account: "Account",
};

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.style.color = isError ? "#9f2f16" : "";
}

function openDrawer() {
  drawer.classList.remove("hidden");
  drawerBackdrop.classList.remove("hidden");
  menuToggleBtn.setAttribute("aria-expanded", "true");
  drawer.setAttribute("aria-hidden", "false");
}

function closeDrawer() {
  drawer.classList.add("hidden");
  drawerBackdrop.classList.add("hidden");
  menuToggleBtn.setAttribute("aria-expanded", "false");
  drawer.setAttribute("aria-hidden", "true");
}

function setActiveNav(viewId) {
  navLinks.forEach((link) => {
    link.classList.toggle("is-active", link.dataset.viewTarget === viewId);
  });
}

function switchView(viewId) {
  viewSections.forEach((section) => {
    section.classList.toggle("is-active", section.dataset.view === viewId);
  });
  setActiveNav(viewId);
  currentViewEl.textContent = VIEW_LABELS[viewId] || "Dashboard";
  closeDrawer();
  if (viewId !== "live" && liveStream) {
    stopLiveDetection();
  }
}

function formatPercent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function formatPercentMaybe(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return "-";
  }
  return formatPercent(num);
}

function formatEpochs(row) {
  const ran = Number(row.training_epochs_ran);
  const target = Number(row.training_epochs_target);
  const hasRan = Number.isFinite(ran);
  const hasTarget = Number.isFinite(target) && target > 0;

  if (hasRan && hasTarget) {
    return `${ran}/${target}`;
  }
  if (hasRan) {
    return `${ran}`;
  }
  if (hasTarget) {
    return `${target}`;
  }
  return "-";
}

function formatSeconds(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${Number(value).toFixed(2)}s`;
}

function renderProbabilities(rows) {
  probabilitiesEl.innerHTML = "";
  rows.forEach((row) => {
    const wrapper = document.createElement("div");
    wrapper.className = "bar";

    const head = document.createElement("div");
    head.className = "bar-head";
    head.innerHTML = `<span>${row.move}</span><span>${formatPercent(row.confidence)}</span>`;

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${Math.max(0, Math.min(100, row.confidence * 100))}%`;
    track.appendChild(fill);

    wrapper.append(head, track);
    probabilitiesEl.appendChild(wrapper);
  });
}

function setFile(file) {
  if (!file) {
    return;
  }
  lastFile = file;
  fileName.textContent = file.name;
  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.load();
}

function setFlexStatus(text, isError = false) {
  if (!flexStatusEl) {
    return;
  }
  flexStatusEl.textContent = text;
  flexStatusEl.style.color = isError ? "#9f2f16" : "";
}

function setFlexFile(file) {
  if (!file || !flexFileNameEl || !flexPreview) {
    return;
  }
  flexLastFile = file;
  flexFileNameEl.textContent = file.name;
  const url = URL.createObjectURL(file);
  flexPreview.src = url;
  flexPreview.load();
}

function resetFlexResultView() {
  if (!flexPredictedMoveEl) {
    return;
  }
  flexPredictedMoveEl.textContent = "-";
  flexQualityEl.textContent = "-";
  flexPoornessEl.textContent = "-";
  flexReadinessEl.textContent = "-";
  flexSessionIdEl.textContent = "-";
  flexJsonEl.textContent = "No result yet.";
}

function populateFlexGoals(goals) {
  if (!flexGoalSelect) {
    return;
  }
  flexGoalSelect.innerHTML = "";
  goals.forEach((goal) => {
    const option = document.createElement("option");
    option.value = goal;
    option.textContent = goal;
    flexGoalSelect.appendChild(option);
  });
}

async function loadFlexMeta() {
  if (!flexGoalSelect) {
    return;
  }

  try {
    const response = await fetch("/api/flex/health", { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "FlexibilityAI metadata unavailable.");
    }
    populateFlexGoals(data.goals || []);
    setFlexStatus("FlexibilityAI ready.");
  } catch (error) {
    populateFlexGoals([]);
    setFlexStatus(error.message || "FlexibilityAI unavailable.", true);
  }
}

function renderFlexResult(result, includeGoal) {
  flexPredictedMoveEl.textContent = result.predicted_movement || "-";
  flexQualityEl.textContent = Number.isFinite(Number(result.quality_score))
    ? `${Number(result.quality_score).toFixed(2)}`
    : "-";
  flexPoornessEl.textContent = Number.isFinite(Number(result.poorness_score))
    ? `${Number(result.poorness_score).toFixed(2)}`
    : "-";

  const readiness = result.goal_feedback?.readiness_score;
  flexReadinessEl.textContent = Number.isFinite(Number(readiness))
    ? `${Number(readiness).toFixed(2)}`
    : "-";
  flexSessionIdEl.textContent = result.session_record_id || "-";
  flexJsonEl.textContent = JSON.stringify(result, null, 2);
  setFlexStatus(includeGoal ? "Goal readiness completed." : "Flexibility analysis completed.");
}

async function runFlexAnalyze(includeGoal) {
  if (!flexLastFile) {
    setFlexStatus("Please upload a flexibility video first.", true);
    return;
  }

  const body = new FormData();
  body.append("video", flexLastFile);

  const athleteIdRaw = (flexAthleteIdInput?.value || "").trim();
  if (athleteIdRaw) {
    body.append("athlete_id", athleteIdRaw);
  }

  if (includeGoal) {
    const goal = (flexGoalSelect?.value || "").trim();
    if (!goal) {
      setFlexStatus("Choose a goal for readiness analysis.", true);
      return;
    }
    body.append("goal", goal);
    const knownScores = (flexKnownScoresInput?.value || "").trim();
    if (knownScores) {
      body.append("known_scores", knownScores);
    }
  }

  setFlexStatus(includeGoal ? "Running flexibility goal analysis..." : "Running flexibility analysis...");

  try {
    const endpoint = includeGoal ? "/api/flex/analyze-goal" : "/api/flex/analyze";
    const response = await fetch(endpoint, { method: "POST", body });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Flexibility analysis failed.");
    }
    renderFlexResult(data, includeGoal);
  } catch (error) {
    setFlexStatus(error.message || "Flexibility analysis failed.", true);
  }
}

function getSelectedModelId() {
  return modelSelect.value || "";
}

function populateModelSelect(models, defaultModelId) {
  modelSelect.innerHTML = "";
  models.forEach((row) => {
    const option = document.createElement("option");
    option.value = row.id;
    option.textContent = `${row.id} (${row.architecture})`;
    modelSelect.appendChild(option);
  });
  if (defaultModelId && models.some((row) => row.id === defaultModelId)) {
    modelSelect.value = defaultModelId;
  }
}

function populateMoveSelect(moves, selectedMove) {
  actualMoveSelect.innerHTML = "";
  const uniqueMoves = [...new Set(moves.filter(Boolean))];
  uniqueMoves.forEach((move) => {
    const option = document.createElement("option");
    option.value = move;
    option.textContent = move;
    actualMoveSelect.appendChild(option);
  });
  if (selectedMove && uniqueMoves.includes(selectedMove)) {
    actualMoveSelect.value = selectedMove;
  }
}

function openTrainingModal(file, prediction, startValue, endValue) {
  if (!file || !prediction) {
    return;
  }
  const fallbackMoves = prediction.labels || [];
  const moves = availableMoves.length ? availableMoves : fallbackMoves;
  populateMoveSelect(moves, prediction.move);

  const clipStart = startValue ?? prediction.selected_segment?.start_sec ?? 0;
  let clipEnd = endValue ?? prediction.selected_segment?.end_sec ?? null;
  if (clipEnd === null || clipEnd === undefined || clipEnd <= clipStart) {
    clipEnd = clipStart + 2.0;
  }

  trainingStartInput.value = Number(clipStart).toFixed(2);
  trainingEndInput.value = Number(clipEnd).toFixed(2);
  trainingPredictedEl.textContent = prediction.move;
  trainingConfidenceEl.textContent = formatPercent(prediction.confidence || 0);
  modelCorrectInput.checked = actualMoveSelect.value === prediction.move;

  const modalUrl = URL.createObjectURL(file);
  trainingPreview.src = modalUrl;
  trainingPreview.onloadedmetadata = () => {
    trainingPreview.currentTime = Number(trainingStartInput.value || 0);
  };
  trainingPreview.ontimeupdate = () => {
    const end = Number(trainingEndInput.value || 0);
    if (Number.isFinite(end) && end > 0 && trainingPreview.currentTime >= end) {
      trainingPreview.pause();
    }
  };

  trainingModal.classList.remove("hidden");
}

function closeTrainingModal() {
  trainingModal.classList.add("hidden");
  trainingPreview.pause();
}

async function sendPredictRequest(file, startRaw, endRaw, modelId) {
  const body = new FormData();
  if (file instanceof File) {
    body.append("video", file);
  } else {
    body.append("video", file, "live_capture.webm");
  }
  if (startRaw) {
    body.append("start_time", startRaw);
  }
  if (endRaw) {
    body.append("end_time", endRaw);
  }
  if (modelId) {
    body.append("model_id", modelId);
  }

  const response = await fetch("/api/predict", { method: "POST", body });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed.");
  }
  return data;
}

function renderBenchmarkTable(models) {
  benchmarkTableEl.innerHTML = "";
  const head = document.createElement("div");
  head.className = "benchmark-row benchmark-head";
  head.innerHTML = "<span>Model</span><span>Epochs</span><span>Accuracy</span><span>Macro F1</span><span>Best Val Acc</span>";
  benchmarkTableEl.appendChild(head);

  // Defensive dedupe by model_id in case old benchmark files contain duplicates.
  const deduped = [];
  const seen = new Set();
  models.forEach((row) => {
    const key = row.model_id || row.path || "";
    if (seen.has(key)) {
      return;
    }
    seen.add(key);
    deduped.push(row);
  });

  deduped.forEach((row) => {
    const item = document.createElement("div");
    item.className = "benchmark-row";
    item.innerHTML = `<span>${row.model_id}</span><span>${formatEpochs(row)}</span><span>${formatPercentMaybe(row.accuracy)}</span><span>${formatPercentMaybe(row.macro_f1)}</span><span>${formatPercentMaybe(row.best_val_accuracy)}</span>`;
    benchmarkTableEl.appendChild(item);
  });
}

async function loadBenchmark() {
  benchmarkStatusEl.textContent = "Loading benchmark...";
  try {
    const response = await fetch("/api/model-benchmark", { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "No benchmark found.");
    }
    benchmarkBestEl.textContent = data.best_model_id || "-";
    renderBenchmarkTable(data.models || []);
    benchmarkStatusEl.textContent = `Updated: ${data.generated_at_utc || "unknown"}`;
  } catch (error) {
    benchmarkBestEl.textContent = "-";
    benchmarkTableEl.innerHTML = "";
    benchmarkStatusEl.textContent = error.message || "Benchmark unavailable.";
  }
}

function getRecorderOptions() {
  const options = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
  ];
  for (const mimeType of options) {
    if (MediaRecorder.isTypeSupported(mimeType)) {
      return { mimeType };
    }
  }
  return {};
}

async function captureLiveChunk(durationMs) {
  if (!liveStream) {
    throw new Error("Live stream not active.");
  }

  return new Promise((resolve, reject) => {
    let chunks = [];
    let recorder;
    try {
      recorder = new MediaRecorder(liveStream, getRecorderOptions());
    } catch (error) {
      reject(new Error("MediaRecorder unsupported in this browser."));
      return;
    }
    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        chunks.push(event.data);
      }
    };
    recorder.onerror = () => reject(new Error("Failed to record live chunk."));
    recorder.onstop = () => resolve(new Blob(chunks, { type: recorder.mimeType || "video/webm" }));
    recorder.start();
    window.setTimeout(() => recorder.stop(), durationMs);
  });
}

async function runLivePredictCycle() {
  if (liveBusy || !liveStream) {
    return;
  }
  liveBusy = true;
  try {
    const blob = await captureLiveChunk(1600);
    const modelId = getSelectedModelId();
    const data = await sendPredictRequest(blob, "", "", modelId);
    liveMoveEl.textContent = data.move;
    liveConfidenceEl.textContent = formatPercent(data.confidence);
    liveStatusEl.textContent = `Live detecting (${data.model_id})`;
  } catch (error) {
    liveStatusEl.textContent = error.message || "Live prediction failed.";
  } finally {
    liveBusy = false;
  }
}

async function startLiveDetection() {
  if (liveStream) {
    return;
  }
  try {
    liveStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    liveVideo.srcObject = liveStream;
    liveStatusEl.textContent = "Camera running...";
    startLiveBtn.disabled = true;
    stopLiveBtn.disabled = false;
    await runLivePredictCycle();
    liveInterval = window.setInterval(runLivePredictCycle, 2300);
  } catch (error) {
    liveStatusEl.textContent = "Could not access camera.";
  }
}

function stopLiveDetection() {
  if (liveInterval) {
    window.clearInterval(liveInterval);
    liveInterval = null;
  }
  if (liveStream) {
    liveStream.getTracks().forEach((track) => track.stop());
    liveStream = null;
  }
  liveVideo.srcObject = null;
  startLiveBtn.disabled = false;
  stopLiveBtn.disabled = true;
  liveStatusEl.textContent = "Camera is off.";
  liveMoveEl.textContent = "-";
  liveConfidenceEl.textContent = "-";
}

input.addEventListener("change", () => {
  const [file] = input.files;
  setFile(file);
});

setStartBtn.addEventListener("click", () => {
  if (Number.isFinite(preview.currentTime)) {
    startTimeInput.value = preview.currentTime.toFixed(2);
  }
});

setEndBtn.addEventListener("click", () => {
  if (Number.isFinite(preview.currentTime)) {
    endTimeInput.value = preview.currentTime.toFixed(2);
  }
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
  });
});

dropZone.addEventListener("drop", (event) => {
  const droppedFiles = event.dataTransfer?.files;
  if (!droppedFiles || !droppedFiles.length) {
    return;
  }
  input.files = droppedFiles;
  setFile(droppedFiles[0]);
});

if (flexVideoInput) {
  flexVideoInput.addEventListener("change", () => {
    const [file] = flexVideoInput.files || [];
    setFlexFile(file);
  });
}

if (flexDropZone && flexVideoInput) {
  ["dragenter", "dragover"].forEach((eventName) => {
    flexDropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      flexDropZone.classList.add("is-dragging");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    flexDropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      flexDropZone.classList.remove("is-dragging");
    });
  });

  flexDropZone.addEventListener("drop", (event) => {
    const droppedFiles = event.dataTransfer?.files;
    if (!droppedFiles || !droppedFiles.length) {
      return;
    }
    flexVideoInput.files = droppedFiles;
    setFlexFile(droppedFiles[0]);
  });
}

if (flexForm) {
  flexForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await runFlexAnalyze(false);
  });
}

if (flexAnalyzeGoalBtn) {
  flexAnalyzeGoalBtn.addEventListener("click", async () => {
    await runFlexAnalyze(true);
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const [file] = input.files;
  if (!file) {
    setStatus("Pick a video file first.", true);
    return;
  }

  const startRaw = startTimeInput.value.trim();
  const endRaw = endTimeInput.value.trim();
  const startValue = startRaw ? Number(startRaw) : null;
  const endValue = endRaw ? Number(endRaw) : null;
  if (startRaw && (!Number.isFinite(startValue) || startValue < 0)) {
    setStatus("Start time must be a number >= 0.", true);
    return;
  }
  if (endRaw && (!Number.isFinite(endValue) || endValue < 0)) {
    setStatus("End time must be a number >= 0.", true);
    return;
  }
  if (startValue !== null && endValue !== null && endValue <= startValue) {
    setStatus("End time must be greater than start time.", true);
    return;
  }

  submitBtn.disabled = true;
  setStatus("Running inference...");
  try {
    const prediction = await sendPredictRequest(file, startRaw, endRaw, getSelectedModelId());
    moveEl.textContent = prediction.move;
    confidenceEl.textContent = formatPercent(prediction.confidence);
    predictedModelEl.textContent = prediction.model_id || "-";
    framesEl.textContent = `${prediction.frames_with_landmarks}`;
    selectedSegmentEl.textContent = `${formatSeconds(prediction.selected_segment?.start_sec)} - ${formatSeconds(prediction.selected_segment?.end_sec)}`;
    bestWindowEl.textContent = `${formatSeconds(prediction.best_window?.start_sec)} - ${formatSeconds(prediction.best_window?.end_sec)}`;
    windowsTestedEl.textContent = `${prediction.windows_evaluated ?? "-"}`;
    renderProbabilities(prediction.probabilities || []);
    setStatus(`Done. Model: ${prediction.model_path}`);
    lastPrediction = prediction;
    openTrainingModal(file, prediction, startValue, endValue);
  } catch (error) {
    setStatus(error.message || "Failed to predict.", true);
  } finally {
    submitBtn.disabled = false;
  }
});

actualMoveSelect.addEventListener("change", () => {
  if (!lastPrediction) {
    return;
  }
  modelCorrectInput.checked = actualMoveSelect.value === lastPrediction.move;
});

cancelTrainingBtn.addEventListener("click", () => {
  closeTrainingModal();
  setStatus("Prediction saved. Training clip was not submitted.");
});

confirmTrainingBtn.addEventListener("click", async () => {
  const file = lastFile;
  const prediction = lastPrediction;
  if (!file || !prediction) {
    setStatus("No prediction available to submit.", true);
    return;
  }

  const clipStart = Number(trainingStartInput.value);
  const clipEnd = Number(trainingEndInput.value);
  if (!Number.isFinite(clipStart) || clipStart < 0) {
    setStatus("Training clip start must be a valid number >= 0.", true);
    return;
  }
  if (!Number.isFinite(clipEnd) || clipEnd <= clipStart) {
    setStatus("Training clip end must be greater than start.", true);
    return;
  }
  if (!actualMoveSelect.value) {
    setStatus("Select the actual move label before sending.", true);
    return;
  }

  const body = new FormData();
  body.append("video", file);
  body.append("start_time", clipStart.toString());
  body.append("end_time", clipEnd.toString());
  body.append("actual_move", actualMoveSelect.value);
  body.append("predicted_move", prediction.move);
  body.append("predicted_confidence", String(prediction.confidence ?? ""));
  body.append("model_correct", modelCorrectInput.checked ? "true" : "false");
  body.append("model_id", prediction.model_id || "");

  confirmTrainingBtn.disabled = true;
  setStatus("Sending selected clip for training...");
  try {
    const response = await fetch("/api/training-sample", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Failed to save training sample.");
    }
    closeTrainingModal();
    if (data.landmarks_saved) {
      setStatus(`Training clip saved: ${data.sample_id}`);
    } else {
      setStatus(`Clip saved (${data.sample_id}), but no landmarks detected.`);
    }
  } catch (error) {
    setStatus(error.message || "Failed to save training sample.", true);
  } finally {
    confirmTrainingBtn.disabled = false;
  }
});

refreshBenchmarkBtn.addEventListener("click", () => {
  loadBenchmark();
});

startLiveBtn.addEventListener("click", () => {
  startLiveDetection();
});

stopLiveBtn.addEventListener("click", () => {
  stopLiveDetection();
});

menuToggleBtn.addEventListener("click", () => {
  openDrawer();
});

drawerCloseBtn.addEventListener("click", () => {
  closeDrawer();
});

drawerBackdrop.addEventListener("click", () => {
  closeDrawer();
});

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeDrawer();
  }
});

topLoginBtn.addEventListener("click", () => {
  switchView("login");
});

navLinks.forEach((link) => {
  link.addEventListener("click", () => {
    switchView(link.dataset.viewTarget || "dashboard");
  });
});

loginForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const email = (loginEmailInput.value || "").trim();
  if (!email) {
    loginStatusEl.textContent = "Enter a valid email.";
    return;
  }
  loginStatusEl.textContent = `Signed in as ${email} (demo mode)`;
  switchView("dashboard");
});

async function loadMoves() {
  try {
    const response = await fetch("/api/moves");
    if (!response.ok) {
      return;
    }
    const data = await response.json();
    const moves = data.moves || [];
    availableMoves = moves;
    chipsEl.innerHTML = "";
    moves.forEach((move) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = move;
      chipsEl.appendChild(chip);
    });
  } catch (_error) {
    // Non-blocking metadata load.
  }
}

async function loadModels() {
  try {
    const response = await fetch("/api/models");
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Failed to load models.");
    }
    availableModels = data.models || [];
    populateModelSelect(availableModels, data.default_model_id);
  } catch (_error) {
    modelSelect.innerHTML = "<option value=''>No models</option>";
  }
}

stopLiveBtn.disabled = true;
loadModels();
loadMoves();
loadBenchmark();
resetFlexResultView();
loadFlexMeta();
