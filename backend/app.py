from __future__ import annotations

import json
import math
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

try:
    from backend.inference import CapoeiraInferenceService
except ModuleNotFoundError:  # Allows `python backend/app.py`
    from inference import CapoeiraInferenceService

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = BASE_DIR / "frontend"
STATIC_DIR = BASE_DIR / "frontend" / "static"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DATA_ROOT = BASE_DIR / "data"
PROCESSED_ROOT = BASE_DIR / "processed_landmarks"
TRAINING_LOG = DATA_ROOT / "training_feedback.jsonl"
BENCHMARK_PATH = BASE_DIR / "models" / "model_benchmark.json"

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
    static_url_path="/static",
)
service = CapoeiraInferenceService(base_dir=BASE_DIR)


def _parse_time_range(start_raw: str, end_raw: str) -> tuple[float | None, float | None, str | None]:
    try:
        start_time = float(start_raw) if start_raw else None
        end_time = float(end_raw) if end_raw else None
    except ValueError:
        return None, None, "start_time and end_time must be numeric seconds."

    if start_time is not None and start_time < 0:
        return None, None, "start_time must be >= 0."
    if end_time is not None and end_time < 0:
        return None, None, "end_time must be >= 0."
    if start_time is not None and end_time is not None and end_time <= start_time:
        return None, None, "end_time must be greater than start_time."

    return start_time, end_time, None


def _normalize_move_label(label: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in label.strip())
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_")


def _save_video_clip(source_path: Path, target_path: Path, start_time: float, end_time: float) -> None:
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise ValueError("Could not open uploaded video for clipping.")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Could not read video dimensions.")

    # Auto-round clip boundaries to frame-safe values:
    # start rounds down, end rounds up.
    start_frame = max(0, math.floor(start_time * fps))
    end_frame = max(0, math.ceil(end_time * fps))
    if end_frame <= start_frame:
        cap.release()
        raise ValueError("Selected clip is empty. Increase end time.")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(target_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    while cap.isOpened() and frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    if not target_path.exists() or target_path.stat().st_size == 0:
        raise ValueError("Failed to create clip file from selected segment.")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "model_path": str(service.model_path.relative_to(BASE_DIR)),
            "default_model_id": service.default_model_id,
            "sequence_length": service.sequence_length,
            "labels": service.labels,
        }
    )


@app.get("/api/moves")
def moves():
    return jsonify({"moves": service.labels})


@app.get("/api/models")
def models():
    return jsonify(
        {
            "default_model_id": service.default_model_id,
            "models": service.get_available_models(),
        }
    )


@app.get("/api/model-benchmark")
def model_benchmark():
    if not BENCHMARK_PATH.exists():
        return jsonify({"error": "No benchmark found. Run 08_compare_models.py first."}), 404
    try:
        with BENCHMARK_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return jsonify({"error": f"Could not read benchmark file: {exc}"}), 500
    response = jsonify(data)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


@app.post("/api/predict")
def predict():
    if "video" not in request.files:
        return jsonify({"error": "Missing file field: video"}), 400

    file = request.files["video"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected."}), 400

    filename = secure_filename(file.filename)
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return (
            jsonify(
                {
                    "error": "Unsupported file type. Use one of: "
                    + ", ".join(sorted(ALLOWED_EXTENSIONS))
                }
            ),
            400,
        )

    start_time_raw = (request.form.get("start_time") or "").strip()
    end_time_raw = (request.form.get("end_time") or "").strip()
    model_id = (request.form.get("model_id") or "").strip() or None
    start_time, end_time, parse_error = _parse_time_range(start_time_raw, end_time_raw)
    if parse_error:
        return jsonify({"error": parse_error}), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
            tmp_path = Path(tmp_file.name)
            file.save(tmp_file.name)

        prediction = service.predict_video(
            tmp_path,
            start_time=start_time,
            end_time=end_time,
            model_id=model_id,
        )
        prediction["filename"] = filename
        return jsonify(prediction)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Prediction failed: {exc}"}), 500
    finally:
        if tmp_path and tmp_path.exists():
            os.remove(tmp_path)


@app.post("/api/training-sample")
def training_sample():
    if "video" not in request.files:
        return jsonify({"error": "Missing file field: video"}), 400

    file = request.files["video"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected."}), 400

    filename = secure_filename(file.filename)
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return (
            jsonify(
                {
                    "error": "Unsupported file type. Use one of: "
                    + ", ".join(sorted(ALLOWED_EXTENSIONS))
                }
            ),
            400,
        )

    start_time_raw = (request.form.get("start_time") or "").strip()
    end_time_raw = (request.form.get("end_time") or "").strip()
    start_time, end_time, parse_error = _parse_time_range(start_time_raw, end_time_raw)
    if parse_error:
        return jsonify({"error": parse_error}), 400
    if start_time is None or end_time is None:
        return jsonify({"error": "Both start_time and end_time are required for training clip save."}), 400

    actual_move = _normalize_move_label(request.form.get("actual_move", ""))
    if not actual_move:
        return jsonify({"error": "actual_move is required."}), 400

    predicted_move = request.form.get("predicted_move", "")
    predicted_confidence = request.form.get("predicted_confidence", "")
    model_id = request.form.get("model_id", "")
    model_correct = str(request.form.get("model_correct", "")).lower() in {"1", "true", "yes", "on"}

    sample_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    clip_path = DATA_ROOT / actual_move / f"{sample_id}.mp4"
    npy_path = PROCESSED_ROOT / actual_move / f"{sample_id}.npy"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
            tmp_path = Path(tmp_file.name)
            file.save(tmp_file.name)

        _save_video_clip(
            source_path=tmp_path,
            target_path=clip_path,
            start_time=start_time,
            end_time=end_time,
        )

        landmarks, _, _ = service.extract_landmarks(clip_path)
        landmarks_saved = False
        if isinstance(landmarks, np.ndarray) and landmarks.size > 0:
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy_path, landmarks)
            landmarks_saved = True

        TRAINING_LOG.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sample_id": sample_id,
            "video_path": str(clip_path.relative_to(BASE_DIR)),
            "landmarks_path": str(npy_path.relative_to(BASE_DIR)) if landmarks_saved else None,
            "actual_move": actual_move,
            "predicted_move": predicted_move,
            "predicted_confidence": predicted_confidence,
            "model_id": model_id,
            "model_correct": model_correct,
            "start_time": start_time,
            "end_time": end_time,
            "landmarks_saved": landmarks_saved,
        }
        with TRAINING_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        return jsonify(
            {
                "ok": True,
                "sample_id": sample_id,
                "video_path": entry["video_path"],
                "landmarks_path": entry["landmarks_path"],
                "landmarks_saved": landmarks_saved,
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Failed to save training sample: {exc}"}), 500
    finally:
        if tmp_path and tmp_path.exists():
            os.remove(tmp_path)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
