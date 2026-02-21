from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

FLEX_BASE_DIR = BASE_DIR / "FlexibilityAI"
FLEX_TEMPLATES_DIR = FLEX_BASE_DIR / "templates"
FLEX_CONFIG_PATH = FLEX_BASE_DIR / "config" / "movements.json"
FLEX_GOALS_CONFIG_PATH = FLEX_BASE_DIR / "config" / "goals.json"
FLEX_YOUTUBE_QUERY_CONFIG_PATH = FLEX_BASE_DIR / "config" / "youtube_queries.json"
FLEX_YT_DOWNLOADS_DIR = FLEX_BASE_DIR / "artifacts" / "youtube_downloads"
FLEX_YT_CLIPS_DIR = FLEX_BASE_DIR / "artifacts" / "youtube_clips"
FLEX_DB_PATH = FLEX_BASE_DIR / "data" / "flexibilityai.db"
FLEX_OPENAI_MODEL = os.getenv("FLEXIBILITYAI_OPENAI_MODEL", "gpt-4o-mini")
FLEX_AVAILABLE = False
FLEX_INIT_ERROR: str | None = None
flex_analyzer = None
flex_store = None
flex_coach = None
flex_bootstrap_templates_from_youtube = None

if FLEX_BASE_DIR.exists():
    if str(FLEX_BASE_DIR) not in sys.path:
        sys.path.insert(0, str(FLEX_BASE_DIR))
    try:
        from flexibilityai.analyzer import MovementAnalyzer
        from flexibilityai.coach import CoachingResponder
        from flexibilityai.storage import FlexibilityStore
        from flexibilityai.youtube import bootstrap_templates_from_youtube

        flex_analyzer = MovementAnalyzer(
            templates_dir=FLEX_TEMPLATES_DIR,
            movement_config_path=FLEX_CONFIG_PATH,
            goals_config_path=FLEX_GOALS_CONFIG_PATH,
        )
        flex_store = FlexibilityStore(db_path=FLEX_DB_PATH)
        flex_coach = CoachingResponder(model=FLEX_OPENAI_MODEL)
        flex_bootstrap_templates_from_youtube = bootstrap_templates_from_youtube
        FLEX_AVAILABLE = True
    except Exception as exc:  # pragma: no cover
        FLEX_INIT_ERROR = str(exc)
else:
    FLEX_INIT_ERROR = f"FlexibilityAI directory not found at {FLEX_BASE_DIR}"


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


def _parse_flex_known_scores(raw: str) -> dict[str, float]:
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"known_scores must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("known_scores must be a JSON object, e.g. {\"bridge\": 72.5}.")
    scores: dict[str, float] = {}
    for key, value in parsed.items():
        try:
            scores[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"known_scores value for '{key}' must be numeric.") from exc
    return scores


def _parse_optional_athlete_id(raw: str) -> int | None:
    value = (raw or "").strip()
    if not value:
        return None
    try:
        athlete_id = int(value)
    except ValueError as exc:
        raise ValueError("athlete_id must be an integer.") from exc
    if athlete_id <= 0:
        raise ValueError("athlete_id must be >= 1.")
    return athlete_id


def _flex_unavailable_response():
    return (
        jsonify(
            {
                "error": "FlexibilityAI is unavailable in this runtime.",
                "details": FLEX_INIT_ERROR,
            }
        ),
        503,
    )


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    payload = {
        "ok": True,
        "model_path": str(service.model_path.relative_to(BASE_DIR)),
        "default_model_id": service.default_model_id,
        "sequence_length": service.sequence_length,
        "labels": service.labels,
        "flexibility_available": FLEX_AVAILABLE,
    }
    if FLEX_AVAILABLE and flex_analyzer is not None:
        payload["flexibility"] = {
            "movements": flex_analyzer.movement_names,
            "goals": flex_analyzer.goal_names,
            "db_path": str(FLEX_DB_PATH.relative_to(BASE_DIR)),
        }
    elif FLEX_INIT_ERROR:
        payload["flexibility_error"] = FLEX_INIT_ERROR
    return jsonify(payload)


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


@app.get("/api/flex/health")
def flex_health():
    if not FLEX_AVAILABLE or flex_analyzer is None:
        return _flex_unavailable_response()
    return jsonify(
        {
            "ok": True,
            "movements": flex_analyzer.movement_names,
            "goals": flex_analyzer.goal_names,
            "template_counts": flex_analyzer.template_counts(),
            "templates_dir": str(FLEX_TEMPLATES_DIR),
            "config_path": str(FLEX_CONFIG_PATH),
            "goals_config_path": str(FLEX_GOALS_CONFIG_PATH),
            "db_path": str(FLEX_DB_PATH),
            "openai_configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
            "openai_model": FLEX_OPENAI_MODEL,
        }
    )


@app.get("/api/flex/moves")
def flex_moves():
    if not FLEX_AVAILABLE or flex_analyzer is None:
        return _flex_unavailable_response()
    return jsonify({"moves": flex_analyzer.movement_names})


@app.get("/api/flex/goals")
def flex_goals():
    if not FLEX_AVAILABLE or flex_analyzer is None:
        return _flex_unavailable_response()
    return jsonify({"goals": flex_analyzer.goal_names})


@app.post("/api/flex/template")
def flex_add_template():
    if not FLEX_AVAILABLE or flex_analyzer is None:
        return _flex_unavailable_response()

    movement = (request.form.get("movement") or "").strip()
    template_name = (request.form.get("template_name") or "").strip() or None

    if movement not in flex_analyzer.movement_names:
        return (
            jsonify(
                {
                    "error": (
                        "Unknown or missing 'movement'. "
                        f"Use one of: {', '.join(flex_analyzer.movement_names)}"
                    )
                }
            ),
            400,
        )

    if "video" not in request.files:
        return jsonify({"error": "Missing file field: video"}), 400
    video = request.files["video"]
    if not video or not video.filename:
        return jsonify({"error": "No video selected."}), 400

    filename = secure_filename(video.filename)
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: {extension}"}), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
            tmp_path = Path(tmp_file.name)
            video.save(tmp_file.name)

        result = flex_analyzer.create_template(
            movement=movement,
            video_path=tmp_path,
            template_name=template_name,
        )
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": str(exc)}), 422
    finally:
        if tmp_path and tmp_path.exists():
            os.remove(tmp_path)


@app.post("/api/flex/analyze")
def flex_analyze():
    if not FLEX_AVAILABLE or flex_analyzer is None:
        return _flex_unavailable_response()

    if "video" not in request.files:
        return jsonify({"error": "Missing file field: video"}), 400
    video = request.files["video"]
    if not video or not video.filename:
        return jsonify({"error": "No video selected."}), 400

    filename = secure_filename(video.filename)
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: {extension}"}), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
            tmp_path = Path(tmp_file.name)
            video.save(tmp_file.name)

        result = flex_analyzer.analyze_video(tmp_path)
        result["filename"] = filename

        athlete_id = _parse_optional_athlete_id(request.form.get("athlete_id") or "")
        if athlete_id is not None:
            if flex_store is None:
                raise ValueError("FlexibilityAI store is unavailable.")
            record = flex_store.add_performance_record(
                athlete_id=athlete_id,
                result=result,
                source_type="analyze",
                filename=filename,
            )
            result["session_record_id"] = int(record["id"])

        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": str(exc)}), 422
    finally:
        if tmp_path and tmp_path.exists():
            os.remove(tmp_path)


@app.post("/api/flex/analyze-goal")
def flex_analyze_goal():
    if not FLEX_AVAILABLE or flex_analyzer is None:
        return _flex_unavailable_response()

    goal = (request.form.get("goal") or "").strip()
    if not goal:
        return jsonify({"error": "Missing required form field: goal"}), 400

    known_scores_raw = request.form.get("known_scores") or ""
    try:
        known_scores = _parse_flex_known_scores(known_scores_raw)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if "video" not in request.files:
        return jsonify({"error": "Missing file field: video"}), 400
    video = request.files["video"]
    if not video or not video.filename:
        return jsonify({"error": "No video selected."}), 400

    filename = secure_filename(video.filename)
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: {extension}"}), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
            tmp_path = Path(tmp_file.name)
            video.save(tmp_file.name)

        result = flex_analyzer.analyze_video_for_goal(
            video_path=tmp_path,
            goal=goal,
            known_scores=known_scores,
        )
        result["filename"] = filename

        athlete_id = _parse_optional_athlete_id(request.form.get("athlete_id") or "")
        if athlete_id is not None:
            if flex_store is None:
                raise ValueError("FlexibilityAI store is unavailable.")
            record = flex_store.add_performance_record(
                athlete_id=athlete_id,
                result=result,
                source_type="analyze_goal",
                goal=goal,
                filename=filename,
            )
            result["session_record_id"] = int(record["id"])

        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": str(exc)}), 422
    finally:
        if tmp_path and tmp_path.exists():
            os.remove(tmp_path)


@app.post("/api/flex/bootstrap-youtube")
def flex_bootstrap_youtube():
    if not FLEX_AVAILABLE or flex_analyzer is None or flex_bootstrap_templates_from_youtube is None:
        return _flex_unavailable_response()

    payload = request.get_json(silent=True) or {}
    try:
        per_movement = int(payload.get("per_movement", 1))
        clip_start = float(payload.get("clip_start", 8.0))
        clip_end = float(payload.get("clip_end", 22.0))
    except (TypeError, ValueError):
        return jsonify({"error": "per_movement, clip_start, and clip_end must be numeric."}), 400

    movements = payload.get("movements")
    if movements is not None and not isinstance(movements, list):
        return jsonify({"error": "movements must be a JSON array if provided."}), 400

    try:
        summary = flex_bootstrap_templates_from_youtube(
            analyzer=flex_analyzer,
            query_config_path=FLEX_YOUTUBE_QUERY_CONFIG_PATH,
            downloads_dir=FLEX_YT_DOWNLOADS_DIR,
            clips_dir=FLEX_YT_CLIPS_DIR,
            per_movement=per_movement,
            clip_start=clip_start,
            clip_end=clip_end,
            movements=movements,
        )
        return jsonify(summary)
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": str(exc)}), 422


@app.post("/api/flex/admin/athletes")
def flex_create_athlete():
    if not FLEX_AVAILABLE or flex_store is None:
        return _flex_unavailable_response()

    payload = request.get_json(silent=True) or {}
    full_name = str(payload.get("full_name") or "").strip()
    user_id = str(payload.get("user_id") or "").strip() or None
    email = str(payload.get("email") or "").strip() or None
    notes = str(payload.get("notes") or "").strip() or None

    if not full_name:
        return jsonify({"error": "full_name is required."}), 400

    try:
        athlete = flex_store.create_athlete(
            full_name=full_name,
            user_id=user_id,
            email=email,
            notes=notes,
        )
        return jsonify({"athlete": athlete}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "user_id already exists."}), 409
    except Exception as exc:
        return jsonify({"error": str(exc)}), 422


@app.get("/api/flex/admin/athletes")
def flex_list_athletes():
    if not FLEX_AVAILABLE or flex_store is None:
        return _flex_unavailable_response()

    try:
        limit = int(request.args.get("limit", "200"))
    except ValueError:
        return jsonify({"error": "limit must be an integer."}), 400

    athletes = flex_store.list_athletes(limit=limit)
    return jsonify({"count": len(athletes), "athletes": athletes})


@app.get("/api/flex/admin/athletes/<int:athlete_id>")
def flex_get_athlete(athlete_id: int):
    if not FLEX_AVAILABLE or flex_store is None:
        return _flex_unavailable_response()

    athlete = flex_store.get_athlete(athlete_id)
    if athlete is None:
        return jsonify({"error": f"Athlete id {athlete_id} not found."}), 404
    return jsonify({"athlete": athlete})


@app.patch("/api/flex/admin/athletes/<int:athlete_id>")
def flex_patch_athlete(athlete_id: int):
    if not FLEX_AVAILABLE or flex_store is None:
        return _flex_unavailable_response()

    payload = request.get_json(silent=True) or {}
    allowed = {"full_name", "user_id", "email", "notes"}
    changes = {key: payload[key] for key in allowed if key in payload}
    if not changes:
        return jsonify({"error": "Provide at least one field: full_name, user_id, email, notes."}), 400

    if "full_name" in changes:
        full_name = str(changes["full_name"] or "").strip()
        if not full_name:
            return jsonify({"error": "full_name cannot be empty."}), 400
        changes["full_name"] = full_name
    if "user_id" in changes:
        changes["user_id"] = str(changes["user_id"] or "").strip() or None
    if "email" in changes:
        changes["email"] = str(changes["email"] or "").strip() or None
    if "notes" in changes:
        changes["notes"] = str(changes["notes"] or "").strip() or None

    try:
        athlete = flex_store.update_athlete(
            athlete_id,
            full_name=changes.get("full_name"),
            user_id=changes.get("user_id"),
            email=changes.get("email"),
            notes=changes.get("notes"),
        )
        return jsonify({"athlete": athlete})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except sqlite3.IntegrityError:
        return jsonify({"error": "user_id already exists."}), 409
    except Exception as exc:
        return jsonify({"error": str(exc)}), 422


@app.get("/api/flex/admin/athletes/<int:athlete_id>/sessions")
def flex_athlete_sessions(athlete_id: int):
    if not FLEX_AVAILABLE or flex_store is None:
        return _flex_unavailable_response()

    if flex_store.get_athlete(athlete_id) is None:
        return jsonify({"error": f"Athlete id {athlete_id} not found."}), 404

    try:
        limit = int(request.args.get("limit", "50"))
    except ValueError:
        return jsonify({"error": "limit must be an integer."}), 400

    sessions = flex_store.list_performance_records(athlete_id, limit=limit)
    return jsonify({"count": len(sessions), "sessions": sessions})


@app.get("/api/flex/admin/athletes/<int:athlete_id>/analytics")
def flex_athlete_analytics(athlete_id: int):
    if not FLEX_AVAILABLE or flex_store is None:
        return _flex_unavailable_response()

    try:
        analytics = flex_store.athlete_analytics(athlete_id)
        return jsonify(analytics)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 422


@app.post("/api/flex/admin/athletes/<int:athlete_id>/coach-response")
def flex_athlete_coach_response(athlete_id: int):
    if not FLEX_AVAILABLE or flex_store is None or flex_coach is None:
        return _flex_unavailable_response()

    athlete = flex_store.get_athlete(athlete_id)
    if athlete is None:
        return jsonify({"error": f"Athlete id {athlete_id} not found."}), 404

    payload = request.get_json(silent=True) or {}
    goal = str(payload.get("goal") or "").strip() or None

    try:
        analytics = flex_store.athlete_analytics(athlete_id)
        if goal is None and analytics.get("latest_session"):
            latest_goal = analytics["latest_session"].get("goal")
            goal = str(latest_goal).strip() if latest_goal else None
        response_data = flex_coach.generate(
            athlete=athlete,
            analytics=analytics,
            goal=goal,
        )
        return jsonify(
            {
                "athlete": athlete,
                "goal": goal,
                "analytics": analytics,
                "coach_response": response_data,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 422


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
