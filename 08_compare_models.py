import json
import time
import argparse
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

MODELS_DIR = Path("models")
OUTPUT_PATH = MODELS_DIR / "model_benchmark.json"
RANDOM_SEED = 42
TEST_RATIO = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark trained models on a holdout split.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("dataset"),
        help="Dataset directory containing X.npy, y.npy, and label_map.json.",
    )
    return parser.parse_args()


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def infer_training_files(model_id: str) -> tuple[Path, Path]:
    lower = model_id.lower()
    if "conv1d" in lower:
        return (
            MODELS_DIR / "training_history_conv1d.json",
            MODELS_DIR / "capoeira_conv1d_train_meta.json",
        )
    if "lstm" in lower:
        return (
            MODELS_DIR / "training_history_lstm.json",
            MODELS_DIR / "capoeira_lstm_train_meta.json",
        )
    return (
        MODELS_DIR / "training_history.json",
        MODELS_DIR / "capoeira_model_train_meta.json",
    )


def load_training_summary(model_id: str) -> dict:
    history_path, meta_path = infer_training_files(model_id)
    history = load_json_if_exists(history_path) or {}
    meta = load_json_if_exists(meta_path) or {}

    loss_hist = history.get("loss") or []
    val_acc_hist = history.get("val_accuracy") or []
    val_loss_hist = history.get("val_loss") or []

    epochs_ran = len(loss_hist) if loss_hist else None
    best_val_accuracy = max(val_acc_hist) if val_acc_hist else None
    best_val_loss = min(val_loss_hist) if val_loss_hist else None
    training_epochs_target = meta.get("epochs")

    return {
        "training_history_path": str(history_path) if history else None,
        "training_meta_path": str(meta_path) if meta else None,
        "training_epochs_target": int(training_epochs_target) if isinstance(training_epochs_target, int) else None,
        "training_epochs_ran": int(epochs_ran) if epochs_ran is not None else None,
        "best_val_accuracy": float(best_val_accuracy) if best_val_accuracy is not None else None,
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "run_tag": meta.get("run_tag"),
    }


def scan_models() -> list[Path]:
    # Prefer .keras when both .keras and .h5 exist for the same model_id.
    # If multiple files of same suffix exist, keep the most recently modified.
    by_model_id: dict[str, Path] = {}
    for pattern in ("capoeira_*.keras", "capoeira_*.h5"):
        for path in sorted(MODELS_DIR.glob(pattern)):
            model_id = path.stem
            current = by_model_id.get(model_id)
            if current is None:
                by_model_id[model_id] = path
                continue

            current_is_keras = current.suffix.lower() == ".keras"
            candidate_is_keras = path.suffix.lower() == ".keras"
            if candidate_is_keras and not current_is_keras:
                by_model_id[model_id] = path
                continue
            if candidate_is_keras == current_is_keras and path.stat().st_mtime > current.stat().st_mtime:
                by_model_id[model_id] = path

    return sorted(by_model_id.values(), key=lambda p: p.stem)


def load_model_labels(model_path: Path, dataset_labels: list[str]) -> list[str]:
    label_path = MODELS_DIR / f"{model_path.stem}_labels.json"
    if not label_path.exists():
        return dataset_labels
    try:
        with label_path.open("r", encoding="utf-8") as f:
            labels = json.load(f)
        return [str(label) for label in labels]
    except Exception:
        return dataset_labels


def infer_architecture(stem: str) -> str:
    lower = stem.lower()
    if "conv1d" in lower:
        return "conv1d"
    if "lstm" in lower:
        return "lstm"
    return "generic"


def stratified_split(y_labels: np.ndarray, test_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []

    for cls in np.unique(y_labels):
        cls_indices = np.where(y_labels == cls)[0]
        rng.shuffle(cls_indices)
        test_count = max(1, int(round(len(cls_indices) * test_ratio)))
        if test_count >= len(cls_indices):
            test_count = max(1, len(cls_indices) - 1)
        test_indices.extend(cls_indices[:test_count].tolist())
        train_indices.extend(cls_indices[test_count:].tolist())

    train_indices = np.array(sorted(train_indices), dtype=np.int32)
    test_indices = np.array(sorted(test_indices), dtype=np.int32)
    return train_indices, test_indices


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    n_classes = len(labels)
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    per_class = []
    for i, label in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(cm[i, :].sum()),
            }
        )

    accuracy = float(np.mean(y_true == y_pred))
    macro_f1 = float(np.mean([row["f1"] for row in per_class])) if per_class else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir

    x = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")
    with (data_dir / "label_map.json").open("r", encoding="utf-8") as f:
        dataset_labels = json.load(f)
    dataset_labels = [str(label) for label in dataset_labels]
    dataset_label_to_idx = {label: idx for idx, label in enumerate(dataset_labels)}

    y_labels_full = np.argmax(y, axis=1)
    _, test_idx = stratified_split(y_labels_full, TEST_RATIO, RANDOM_SEED)
    x_test = x[test_idx]
    y_test_full = y_labels_full[test_idx]

    model_paths = scan_models()
    if not model_paths:
        raise FileNotFoundError("No model files found in models/.")

    results = []
    for model_path in model_paths:
        model = load_model(model_path)
        model_labels = load_model_labels(model_path, dataset_labels)
        train_summary = load_training_summary(model_path.stem)
        model_output_dim = int(model.output_shape[-1])
        if len(model_labels) != model_output_dim:
            # Fallback to generic class names when labels are unavailable/mismatched.
            model_labels = [f"class_{i}" for i in range(model_output_dim)]

        # Map model labels back to dataset class ids when possible.
        known_label_pairs = [
            (dataset_label_to_idx[label], idx)
            for idx, label in enumerate(model_labels)
            if label in dataset_label_to_idx
        ]
        if known_label_pairs:
            allowed_dataset_ids = {pair[0] for pair in known_label_pairs}
            mask = np.array([cls in allowed_dataset_ids for cls in y_test_full], dtype=bool)
            x_eval = x_test[mask]
            y_eval_full = y_test_full[mask]
            full_to_model = {full: model_i for full, model_i in known_label_pairs}
            y_eval = np.array([full_to_model[int(cls)] for cls in y_eval_full], dtype=np.int32)
        else:
            # If labels are unknown, evaluate against shared index space.
            x_eval = x_test
            y_eval = np.clip(y_test_full, 0, model_output_dim - 1).astype(np.int32)

        if len(x_eval) == 0:
            continue

        t0 = time.perf_counter()
        preds = model.predict(x_eval, batch_size=16, verbose=0)
        elapsed = time.perf_counter() - t0

        y_pred = np.argmax(preds, axis=1)
        metrics = compute_metrics(y_eval, y_pred, model_labels)
        ms_per_sample = (elapsed / len(x_eval) * 1000.0) if len(x_eval) else 0.0

        results.append(
            {
                "model_id": model_path.stem,
                "path": str(model_path),
                "architecture": infer_architecture(model_path.stem),
                "num_test_samples": int(len(x_eval)),
                "label_count": int(len(model_labels)),
                "inference_ms_per_sample": ms_per_sample,
                **train_summary,
                **metrics,
            }
        )

    if not results:
        raise RuntimeError("No benchmark results were produced. Check model files and labels.")

    results.sort(key=lambda row: row["accuracy"], reverse=True)
    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split": {
            "method": "stratified_holdout",
            "test_ratio": TEST_RATIO,
            "seed": RANDOM_SEED,
            "num_test_samples": int(len(x_test)),
        },
        "models": results,
        "best_model_id": results[0]["model_id"],
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved model comparison to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
