# CapoAI

CapoAI is an AI movement coaching platform focused on Capoeira training.  
It uses computer vision to recognize moves from video and is designed to evolve into a broader AI feedback system for form correction, mobility, and personalized training progression.

## Executive Summary

- Product: `CapoAI` (working title)
- Core goal: help users improve technique (for example `ginga`, `au`, `meia_lua_de_frente`, `queixada`, and others) through AI-based analysis
- Current implementation: video-to-move classification pipeline + Flask web app for predictions
- Target direction: richer form feedback and adaptive coaching workflows

## Project Phases

### Phase 1: Data Collection and Landmark Extraction

Collect movement videos and extract pose landmarks with MediaPipe.

```bash
python 00_download_and_process_cc_videos.py --moves all --per-query 25 --max-per-move 150
```

Example for newer move set:

```bash
python 00_download_and_process_cc_videos.py --moves armada,cocorinha,esquiva_lateral,esquiva_atras,esquiva_baixa,negativa --per-query 40 --max-per-move 250
```

Outputs:
- `data/<move>/` (videos)
- `processed_landmarks/<move>/*.npy` (landmarks)
- `data/collection_report.json` (run report)

### Phase 2: Dataset Build

Convert extracted landmarks into training tensors.

```bash
python 02_preprocess_data.py
```

Outputs:
- `dataset/X.npy`
- `dataset/y.npy`
- `dataset/label_map.json`

### Phase 3: Model Training

Train the sequence model for move classification.

```bash
python 03_train_model.py
```

Outputs:
- `models/capoeira_model_best.keras` (or `.h5`)
- `models/capoeira_model_final.keras`
- `models/training_history.json`

## Quick Environment Setup (Apple Silicon)

```bash
cd /Users/jevanteqaiyim/Desktop/CapoAI
conda deactivate
export CONDA_SUBDIR=osx-arm64
conda create -n capoai-arm64 python=3.10 -y
conda activate capoai-arm64
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import platform; print(platform.machine())"
```

Expected architecture output: `arm64`

### Phase 4: Run the Training App (Web Inference App)

Start the backend app with:

```bash
python -m backend.app
```

Open:
- `http://127.0.0.1:5000`

API endpoints:
- `GET /api/health`
- `GET /api/moves`
- `POST /api/predict` (form-data field: `video`)

### Phase 5: Next Product Expansion (Planned)

- Form-quality feedback (alignment, rhythm, balance cues)
- Mobility/flexibility scoring
- Adaptive training plan generation
- Expanded athlete and martial arts use cases

## Optional Scripts

- Single video prediction: `python 05_predict_move.py`
- Real-time webcam inference: `python 06_realtime_feedback.py`
# CapoAI
