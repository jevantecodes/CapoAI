from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from flexibilityai.analyzer import MovementAnalyzer
from flexibilityai.coach import CoachingResponder
from flexibilityai.storage import FlexibilityStore
from flexibilityai.youtube import bootstrap_templates_from_youtube

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATES_DIR = BASE_DIR / "templates"
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "movements.json"
DEFAULT_GOALS_CONFIG_PATH = BASE_DIR / "config" / "goals.json"
DEFAULT_YOUTUBE_QUERY_CONFIG_PATH = BASE_DIR / "config" / "youtube_queries.json"
DEFAULT_YT_DOWNLOADS_DIR = BASE_DIR / "artifacts" / "youtube_downloads"
DEFAULT_YT_CLIPS_DIR = BASE_DIR / "artifacts" / "youtube_clips"
DEFAULT_DB_PATH = BASE_DIR / "data" / "flexibilityai.db"
DEFAULT_OPENAI_MODEL = os.getenv("FLEXIBILITYAI_OPENAI_MODEL", "gpt-4o-mini")
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def create_analyzer(templates_dir: Path, config_path: Path, goals_config_path: Path) -> MovementAnalyzer:
    return MovementAnalyzer(
        templates_dir=templates_dir,
        movement_config_path=config_path,
        goals_config_path=goals_config_path,
    )


def _parse_known_scores(raw: str) -> dict[str, float]:
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


def create_app(
    templates_dir: Path = DEFAULT_TEMPLATES_DIR,
    config_path: Path = DEFAULT_CONFIG_PATH,
    goals_config_path: Path = DEFAULT_GOALS_CONFIG_PATH,
    db_path: Path = DEFAULT_DB_PATH,
    openai_model: str = DEFAULT_OPENAI_MODEL,
) -> Flask:
    app = Flask(__name__)
    analyzer = create_analyzer(
        templates_dir=templates_dir,
        config_path=config_path,
        goals_config_path=goals_config_path,
    )
    store = FlexibilityStore(db_path=db_path)
    coach_responder = CoachingResponder(model=openai_model)

    def _persist_record_if_requested(
        *,
        athlete_id_raw: str,
        result: dict[str, Any],
        source_type: str,
        goal: str | None,
        filename: str | None,
    ) -> int | None:
        athlete_id = _parse_optional_athlete_id(athlete_id_raw)
        if athlete_id is None:
            return None
        record = store.add_performance_record(
            athlete_id=athlete_id,
            result=result,
            source_type=source_type,
            goal=goal,
            filename=filename,
        )
        return int(record["id"])

    @app.get("/api/health")
    def health() -> tuple[str, int] | tuple[dict, int] | dict:
        return jsonify(
            {
                "ok": True,
                "movements": analyzer.movement_names,
                "goals": analyzer.goal_names,
                "template_counts": analyzer.template_counts(),
                "templates_dir": str(analyzer.templates_dir),
                "config_path": str(analyzer.movement_config_path),
                "goals_config_path": str(goals_config_path),
                "db_path": str(db_path),
                "openai_configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
                "openai_model": openai_model,
            }
        )

    @app.post("/api/template")
    def add_template() -> tuple[dict, int] | dict:
        movement = (request.form.get("movement") or "").strip()
        template_name = (request.form.get("template_name") or "").strip() or None

        if movement not in analyzer.movement_names:
            return (
                jsonify(
                    {
                        "error": (
                            "Unknown or missing 'movement'. "
                            f"Use one of: {', '.join(analyzer.movement_names)}"
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
            return (
                jsonify({"error": f"Unsupported file type: {extension}"}),
                400,
            )

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
                temp_path = Path(tmp.name)
                video.save(tmp.name)

            result = analyzer.create_template(
                movement=movement,
                video_path=temp_path,
                template_name=template_name,
            )
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 422
        finally:
            if temp_path and temp_path.exists():
                os.remove(temp_path)

    @app.post("/api/analyze")
    def analyze() -> tuple[dict, int] | dict:
        if "video" not in request.files:
            return jsonify({"error": "Missing file field: video"}), 400

        video = request.files["video"]
        if not video or not video.filename:
            return jsonify({"error": "No video selected."}), 400

        filename = secure_filename(video.filename)
        extension = Path(filename).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f"Unsupported file type: {extension}"}), 400

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
                temp_path = Path(tmp.name)
                video.save(tmp.name)

            result = analyzer.analyze_video(temp_path)
            result["filename"] = filename

            session_record_id = _persist_record_if_requested(
                athlete_id_raw=request.form.get("athlete_id") or "",
                result=result,
                source_type="analyze",
                goal=None,
                filename=filename,
            )
            if session_record_id is not None:
                result["session_record_id"] = session_record_id

            return jsonify(result)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": str(exc)}), 422
        finally:
            if temp_path and temp_path.exists():
                os.remove(temp_path)

    @app.post("/api/analyze-goal")
    def analyze_goal() -> tuple[dict, int] | dict:
        goal = (request.form.get("goal") or "").strip()
        known_scores_raw = request.form.get("known_scores") or ""
        if not goal:
            return jsonify({"error": "Missing required form field: goal"}), 400

        try:
            known_scores = _parse_known_scores(known_scores_raw)
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

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
                temp_path = Path(tmp.name)
                video.save(tmp.name)

            result = analyzer.analyze_video_for_goal(
                video_path=temp_path,
                goal=goal,
                known_scores=known_scores,
            )
            result["filename"] = filename

            session_record_id = _persist_record_if_requested(
                athlete_id_raw=request.form.get("athlete_id") or "",
                result=result,
                source_type="analyze_goal",
                goal=goal,
                filename=filename,
            )
            if session_record_id is not None:
                result["session_record_id"] = session_record_id

            return jsonify(result)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": str(exc)}), 422
        finally:
            if temp_path and temp_path.exists():
                os.remove(temp_path)

    @app.post("/api/bootstrap-youtube")
    def bootstrap_youtube() -> tuple[dict, int] | dict:
        raw = request.get_json(silent=True) or {}
        try:
            per_movement = int(raw.get("per_movement", 1))
            clip_start = float(raw.get("clip_start", 8.0))
            clip_end = float(raw.get("clip_end", 22.0))
        except (TypeError, ValueError):
            return jsonify({"error": "per_movement, clip_start, and clip_end must be numeric."}), 400

        movements = raw.get("movements")

        if movements is not None and not isinstance(movements, list):
            return jsonify({"error": "movements must be a JSON array if provided."}), 400

        try:
            summary = bootstrap_templates_from_youtube(
                analyzer=analyzer,
                query_config_path=DEFAULT_YOUTUBE_QUERY_CONFIG_PATH,
                downloads_dir=DEFAULT_YT_DOWNLOADS_DIR,
                clips_dir=DEFAULT_YT_CLIPS_DIR,
                per_movement=per_movement,
                clip_start=clip_start,
                clip_end=clip_end,
                movements=movements,
            )
            return jsonify(summary)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 422

    @app.post("/api/admin/athletes")
    def create_athlete() -> tuple[dict, int] | dict:
        payload = request.get_json(silent=True) or {}
        full_name = str(payload.get("full_name") or "").strip()
        user_id = str(payload.get("user_id") or "").strip() or None
        email = str(payload.get("email") or "").strip() or None
        notes = str(payload.get("notes") or "").strip() or None

        if not full_name:
            return jsonify({"error": "full_name is required."}), 400

        try:
            athlete = store.create_athlete(
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

    @app.get("/api/admin/athletes")
    def list_athletes() -> tuple[dict, int] | dict:
        try:
            limit = int(request.args.get("limit", "200"))
        except ValueError:
            return jsonify({"error": "limit must be an integer."}), 400

        athletes = store.list_athletes(limit=limit)
        return jsonify({"count": len(athletes), "athletes": athletes})

    @app.get("/api/admin/athletes/<int:athlete_id>")
    def get_athlete(athlete_id: int) -> tuple[dict, int] | dict:
        athlete = store.get_athlete(athlete_id)
        if athlete is None:
            return jsonify({"error": f"Athlete id {athlete_id} not found."}), 404
        return jsonify({"athlete": athlete})

    @app.patch("/api/admin/athletes/<int:athlete_id>")
    def patch_athlete(athlete_id: int) -> tuple[dict, int] | dict:
        payload = request.get_json(silent=True) or {}
        allowed = {"full_name", "user_id", "email", "notes"}
        changes = {k: payload[k] for k in allowed if k in payload}
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
            athlete = store.update_athlete(
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

    @app.get("/api/admin/athletes/<int:athlete_id>/sessions")
    def athlete_sessions(athlete_id: int) -> tuple[dict, int] | dict:
        if store.get_athlete(athlete_id) is None:
            return jsonify({"error": f"Athlete id {athlete_id} not found."}), 404

        try:
            limit = int(request.args.get("limit", "50"))
        except ValueError:
            return jsonify({"error": "limit must be an integer."}), 400

        sessions = store.list_performance_records(athlete_id, limit=limit)
        return jsonify({"count": len(sessions), "sessions": sessions})

    @app.get("/api/admin/athletes/<int:athlete_id>/analytics")
    def athlete_analytics(athlete_id: int) -> tuple[dict, int] | dict:
        try:
            analytics = store.athlete_analytics(athlete_id)
            return jsonify(analytics)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:
            return jsonify({"error": str(exc)}), 422

    @app.post("/api/admin/athletes/<int:athlete_id>/coach-response")
    def athlete_coach_response(athlete_id: int) -> tuple[dict, int] | dict:
        athlete = store.get_athlete(athlete_id)
        if athlete is None:
            return jsonify({"error": f"Athlete id {athlete_id} not found."}), 404

        payload = request.get_json(silent=True) or {}
        goal = str(payload.get("goal") or "").strip() or None

        try:
            analytics = store.athlete_analytics(athlete_id)
            if goal is None and analytics.get("latest_session"):
                latest_goal = analytics["latest_session"].get("goal")
                goal = str(latest_goal).strip() if latest_goal else None
            response_data = coach_responder.generate(
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

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FlexibilityAI baseline for movement detection + quality scoring "
            "against good examples."
        )
    )
    parser.add_argument("--templates-dir", type=Path, default=DEFAULT_TEMPLATES_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--goals-config", type=Path, default=DEFAULT_GOALS_CONFIG_PATH)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)

    sub = parser.add_subparsers(dest="command", required=True)

    serve_parser = sub.add_parser("serve", help="Run Flask API server.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=5010)

    build_parser = sub.add_parser("build-template", help="Register one good example template.")
    build_parser.add_argument("--movement", required=True)
    build_parser.add_argument("--video", required=True, type=Path)
    build_parser.add_argument("--name", default="")

    analyze_parser = sub.add_parser("analyze", help="Detect movement and compute quality score.")
    analyze_parser.add_argument("--video", required=True, type=Path)
    analyze_parser.add_argument("--athlete-id", type=int, default=None)

    analyze_goal_parser = sub.add_parser(
        "analyze-goal",
        help="Detect movement and score readiness for a capoeira goal.",
    )
    analyze_goal_parser.add_argument("--video", required=True, type=Path)
    analyze_goal_parser.add_argument("--goal", required=True)
    analyze_goal_parser.add_argument("--athlete-id", type=int, default=None)
    analyze_goal_parser.add_argument(
        "--known-scores",
        default="{}",
        help='JSON object of prior movement scores, e.g. {"bridge": 72, "lunge_stretch": 68}',
    )

    yt_parser = sub.add_parser(
        "bootstrap-youtube",
        help="Search YouTube, download clips, and build templates automatically.",
    )
    yt_parser.add_argument("--per-movement", type=int, default=1)
    yt_parser.add_argument("--clip-start", type=float, default=8.0)
    yt_parser.add_argument("--clip-end", type=float, default=22.0)
    yt_parser.add_argument(
        "--movements",
        nargs="*",
        default=[],
        help="Optional subset of movements, e.g. bridge deep_squat",
    )
    yt_parser.add_argument("--queries-config", type=Path, default=DEFAULT_YOUTUBE_QUERY_CONFIG_PATH)
    yt_parser.add_argument("--downloads-dir", type=Path, default=DEFAULT_YT_DOWNLOADS_DIR)
    yt_parser.add_argument("--clips-dir", type=Path, default=DEFAULT_YT_CLIPS_DIR)

    create_athlete_parser = sub.add_parser("create-athlete", help="Create athlete profile in admin DB.")
    create_athlete_parser.add_argument("--name", required=True)
    create_athlete_parser.add_argument("--user-id", default="")
    create_athlete_parser.add_argument("--email", default="")
    create_athlete_parser.add_argument("--notes", default="")

    list_athletes_parser = sub.add_parser("list-athletes", help="List athletes in admin DB.")
    list_athletes_parser.add_argument("--limit", type=int, default=200)

    analytics_parser = sub.add_parser("athlete-analytics", help="View athlete analytics summary.")
    analytics_parser.add_argument("--athlete-id", type=int, required=True)

    coach_parser = sub.add_parser("coach-response", help="Generate coaching response for one athlete.")
    coach_parser.add_argument("--athlete-id", type=int, required=True)
    coach_parser.add_argument("--goal", default="")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyzer = create_analyzer(args.templates_dir, args.config, args.goals_config)
    store = FlexibilityStore(db_path=args.db_path)
    coach_responder = CoachingResponder(model=args.openai_model)

    if args.command == "build-template":
        result = analyzer.create_template(
            movement=args.movement,
            video_path=args.video,
            template_name=args.name or None,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "analyze":
        result = analyzer.analyze_video(args.video)
        if args.athlete_id is not None:
            record = store.add_performance_record(
                athlete_id=args.athlete_id,
                result=result,
                source_type="analyze",
                filename=args.video.name,
            )
            result["session_record_id"] = int(record["id"])
        print(json.dumps(result, indent=2))
        return

    if args.command == "analyze-goal":
        result = analyzer.analyze_video_for_goal(
            video_path=args.video,
            goal=args.goal,
            known_scores=_parse_known_scores(args.known_scores),
        )
        if args.athlete_id is not None:
            record = store.add_performance_record(
                athlete_id=args.athlete_id,
                result=result,
                source_type="analyze_goal",
                goal=args.goal,
                filename=args.video.name,
            )
            result["session_record_id"] = int(record["id"])
        print(json.dumps(result, indent=2))
        return

    if args.command == "bootstrap-youtube":
        summary = bootstrap_templates_from_youtube(
            analyzer=analyzer,
            query_config_path=args.queries_config,
            downloads_dir=args.downloads_dir,
            clips_dir=args.clips_dir,
            per_movement=args.per_movement,
            clip_start=args.clip_start,
            clip_end=args.clip_end,
            movements=args.movements or None,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "create-athlete":
        athlete = store.create_athlete(
            full_name=args.name,
            user_id=(args.user_id.strip() or None),
            email=(args.email.strip() or None),
            notes=(args.notes.strip() or None),
        )
        print(json.dumps({"athlete": athlete}, indent=2))
        return

    if args.command == "list-athletes":
        athletes = store.list_athletes(limit=args.limit)
        print(json.dumps({"count": len(athletes), "athletes": athletes}, indent=2))
        return

    if args.command == "athlete-analytics":
        analytics = store.athlete_analytics(args.athlete_id)
        print(json.dumps(analytics, indent=2))
        return

    if args.command == "coach-response":
        athlete = store.get_athlete(args.athlete_id)
        if athlete is None:
            raise ValueError(f"Athlete id {args.athlete_id} not found.")
        analytics = store.athlete_analytics(args.athlete_id)
        goal = args.goal.strip() or None
        response_data = coach_responder.generate(
            athlete=athlete,
            analytics=analytics,
            goal=goal,
        )
        print(
            json.dumps(
                {
                    "athlete": athlete,
                    "goal": goal,
                    "analytics": analytics,
                    "coach_response": response_data,
                },
                indent=2,
            )
        )
        return

    if args.command == "serve":
        app = create_app(
            args.templates_dir,
            args.config,
            args.goals_config,
            args.db_path,
            args.openai_model,
        )
        app.run(host=args.host, port=args.port, debug=True)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
