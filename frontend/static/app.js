const form = document.getElementById("predict-form");
const input = document.getElementById("video-input");
const dropZone = document.getElementById("drop-zone");
const submitBtn = document.getElementById("submit-btn");
const fileName = document.getElementById("file-name");
const preview = document.getElementById("video-preview");
const statusEl = document.getElementById("status");
const moveEl = document.getElementById("predicted-move");
const confidenceEl = document.getElementById("predicted-confidence");
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
let lastPrediction = null;
let lastFile = null;

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.style.color = isError ? "#9f2f16" : "";
}

function formatPercent(value) {
  return `${(value * 100).toFixed(2)}%`;
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

input.addEventListener("change", () => {
  const [file] = input.files;
  setFile(file);
});

setStartBtn.addEventListener("click", () => {
  if (!Number.isFinite(preview.currentTime)) {
    return;
  }
  startTimeInput.value = preview.currentTime.toFixed(2);
});

setEndBtn.addEventListener("click", () => {
  if (!Number.isFinite(preview.currentTime)) {
    return;
  }
  endTimeInput.value = preview.currentTime.toFixed(2);
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

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const [file] = input.files;
  if (!file) {
    setStatus("Pick a video file first.", true);
    return;
  }

  const body = new FormData();
  body.append("video", file);
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

  if (startRaw) {
    body.append("start_time", startRaw);
  }
  if (endRaw) {
    body.append("end_time", endRaw);
  }

  submitBtn.disabled = true;
  setStatus("Running inference...");

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body,
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Request failed.");
    }

    moveEl.textContent = data.move;
    confidenceEl.textContent = formatPercent(data.confidence);
    framesEl.textContent = `${data.frames_with_landmarks}`;
    selectedSegmentEl.textContent = `${formatSeconds(data.selected_segment?.start_sec)} - ${formatSeconds(data.selected_segment?.end_sec)}`;
    bestWindowEl.textContent = `${formatSeconds(data.best_window?.start_sec)} - ${formatSeconds(data.best_window?.end_sec)}`;
    windowsTestedEl.textContent = `${data.windows_evaluated ?? "-"}`;
    renderProbabilities(data.probabilities || []);
    setStatus(`Done. Model: ${data.model_path}`);
    lastPrediction = data;
    openTrainingModal(file, data, startValue, endValue);
  } catch (error) {
    setStatus(error.message || "Failed to predict.", true);
  } finally {
    submitBtn.disabled = false;
  }
});

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

  confirmTrainingBtn.disabled = true;
  setStatus("Sending selected clip for training...");
  try {
    const response = await fetch("/api/training-sample", {
      method: "POST",
      body,
    });
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
  } catch (error) {
    // Non-blocking if metadata endpoint is unavailable.
  }
}

loadMoves();
