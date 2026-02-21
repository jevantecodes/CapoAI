# FlexibilityAI (Flexibility Baseline)

This is a standalone sub-app in CapoAI for:
1. Detecting which movement a user is doing (starter set of 4 flexibility movements)
2. Scoring how well they perform it vs a **good example** template
3. Estimating readiness for capoeira goals (`macaco`, `macaquinho`, `au`) and what needs improvement
4. Managing athlete records + performance trends for admin
5. Generating GPT-style coaching feedback from athlete analytics

The score output includes:
- `quality_score` (0-100): higher is better
- `poorness_score` (0-100): higher is worse (`100 - quality_score`)

## Starter Movements
- `deep_squat`
- `forward_fold`
- `lunge_stretch`
- `bridge`

You can extend movement definitions in:
- `config/movements.json`

## Install
```bash
cd /Users/jevanteqaiyim/Desktop/CapoAI/FlexibilityAI
pip install -r requirements.txt
```

## Step 1: Register Good Examples (Templates)
Capture at least one strong clip for each movement.

```bash
cd /Users/jevanteqaiyim/Desktop/CapoAI/FlexibilityAI

python app.py build-template --movement deep_squat --video /absolute/path/good_squat.mp4 --name coach_squat_1
python app.py build-template --movement forward_fold --video /absolute/path/good_forward_fold.mp4 --name coach_fold_1
python app.py build-template --movement lunge_stretch --video /absolute/path/good_lunge.mp4 --name coach_lunge_1
python app.py build-template --movement bridge --video /absolute/path/good_bridge.mp4 --name coach_bridge_1
```

Templates are saved in:
- `templates/<movement>/<template_name>.npz`

## Optional: Auto-Build Templates from YouTube
This searches YouTube for each movement, downloads videos, clips the selected time window, then builds templates.

```bash
python app.py bootstrap-youtube --per-movement 2 --clip-start 8 --clip-end 22
```

Notes:
- Query terms are in `config/youtube_queries.json`
- Downloaded files go to `artifacts/youtube_downloads/`
- Extracted clips go to `artifacts/youtube_clips/`

## Step 2: Analyze a New Clip
```bash
python app.py analyze --video /absolute/path/user_clip.mp4
```

## Step 3: Get Goal Readiness Feedback (Macaco / Macaquinho / Au)
Use known prior scores plus the current upload to estimate how much improvement is needed.

```bash
python app.py analyze-goal \
  --video /absolute/path/user_clip.mp4 \
  --goal macaco \
  --known-scores '{"bridge": 72, "lunge_stretch": 68, "forward_fold": 66, "deep_squat": 70}'
```

You will get:
- `readiness_score` (0-100)
- `is_ready` vs threshold
- per-movement `gap_to_target`
- prioritized `next_focus` movements

## Step 4: Admin Data Management
Create and manage athletes:

```bash
python app.py create-athlete --name "John Smith" --user-id "1" --email "john@example.com"
python app.py list-athletes
python app.py athlete-analytics --athlete-id 1
```

Attach sessions to athletes when analyzing:

```bash
python app.py analyze --video /absolute/path/user_clip.mp4 --athlete-id 1
python app.py analyze-goal --video /absolute/path/user_clip.mp4 --goal macaco --athlete-id 1
```

## Step 5: GPT Coaching Response From Analytics
Set API key (optional). If missing, app uses built-in fallback coaching text.

```bash
export OPENAI_API_KEY="your_key_here"
python app.py coach-response --athlete-id 1 --goal macaco
```

## Optional: Run API Server
```bash
python app.py serve --host 127.0.0.1 --port 5010
```

Endpoints:
- `GET /api/health`
- `POST /api/template` (multipart: `video`, form: `movement`, optional `template_name`)
- `POST /api/analyze` (multipart: `video`, optional form: `athlete_id`)
- `POST /api/analyze-goal` (multipart: `video`, form: `goal`, optional `known_scores` JSON string, optional `athlete_id`)
- `POST /api/bootstrap-youtube` (JSON body: `per_movement`, `clip_start`, `clip_end`, optional `movements`)
- `POST /api/admin/athletes`
- `GET /api/admin/athletes`
- `GET /api/admin/athletes/<athlete_id>`
- `PATCH /api/admin/athletes/<athlete_id>`
- `GET /api/admin/athletes/<athlete_id>/sessions`
- `GET /api/admin/athletes/<athlete_id>/analytics`
- `POST /api/admin/athletes/<athlete_id>/coach-response`

## How Scoring Works
For the predicted movement, the app compares your clip to the best matching good-example template using:
- DTW sequence similarity
- range-of-motion comparison over movement-specific features
- posture/shape deviation after time normalization

Weighted combination (default in config):
- similarity: `0.65`
- ROM: `0.20`
- posture: `0.15`

## Extend It
- Add new movements in `config/movements.json`
- Adjust capoeira progression thresholds in `config/goals.json`
- Refine YouTube search prompts in `config/youtube_queries.json`
- Record more good templates per movement
- Add a frontend admin dashboard on top of the new admin APIs
