# CapoAI

## About CapoAI
CapoAI is an AI movement-classification app for capoeira training.  
It analyzes pose landmarks from video clips, predicts the movement class, and helps compare model performance across architectures like LSTM and Conv1D.


### Phase 1: Data Collection and Landmark Extraction

Collect movement videos and extract pose landmarks with MediaPipe.

## How To Generate Fresh Data
```bash
cd /Users/jevanteqaiyim/Desktop/CapoAI
conda activate capoai-arm64

python 13_expand_data_pipeline.py \
  --run-tag v2_fresh \
  --moves all \
  --per-query 40 \
  --max-per-move 300 \
  --rebuild-dataset
```

## How To Train The Model
```bash
python 03_train_model.py --data-path dataset/v2_fresh --arch lstm --epochs 120 --batch-size 8
python 03_train_model.py --data-path dataset/v2_fresh --arch conv1d --epochs 120 --batch-size 8
python 08_compare_models.py --data-dir dataset/v2_fresh
```

## How To Run The App
```bash
cd /Users/jevanteqaiyim/Desktop/CapoAI
conda activate capoai-arm64
python -m backend.app
```

Open:
- `http://127.0.0.1:5000`
