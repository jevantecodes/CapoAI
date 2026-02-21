import argparse
from datetime import datetime, timezone
import json
import os
import shutil

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D, Dense, Dropout, GlobalAveragePooling1D, LSTM, LayerNormalization
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CapoAI sequence model.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="dataset",
        help="Dataset directory containing X.npy, y.npy, and label_map.json.",
    )
    parser.add_argument(
        "--arch",
        choices=["conv1d", "lstm"],
        default="conv1d",
        help="Model architecture to train.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size.")
    parser.add_argument(
        "--min-class-samples",
        type=int,
        default=0,
        help=(
            "Exclude classes with fewer than this many samples from this training run. "
            "Data files are not deleted."
        ),
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.25,
        help="Validation ratio for stratified split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified split.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Label smoothing factor for categorical cross-entropy.",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable inverse-frequency class weights.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help=(
            "Optional run tag for versioned artifacts. "
            "If empty, uses UTC timestamp (YYYYMMDD_HHMMSS)."
        ),
    )
    return parser.parse_args()


def build_model(arch: str, seq_len: int, feature_count: int, label_count: int) -> Sequential:
    if arch == "lstm":
        return Sequential(
            [
                LayerNormalization(input_shape=(seq_len, feature_count)),
                LSTM(96, return_sequences=True, activation="tanh"),
                Dropout(0.25),
                LSTM(48, return_sequences=False, activation="tanh"),
                Dense(64, activation="relu"),
                Dropout(0.3),
                Dense(label_count, activation="softmax"),
            ]
        )

    return Sequential(
        [
            LayerNormalization(input_shape=(seq_len, feature_count)),
            Conv1D(64, kernel_size=5, padding="same", activation="relu"),
            Dropout(0.2),
            Conv1D(64, kernel_size=5, padding="same", activation="relu"),
            GlobalAveragePooling1D(),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(label_count, activation="softmax"),
        ]
    )


def filter_underrepresented_classes(
    x: np.ndarray,
    y: np.ndarray,
    moves: list[str],
    min_class_samples: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    y_idx = np.argmax(y, axis=1)
    counts = np.bincount(y_idx, minlength=y.shape[1]).astype(int)

    keep_ids = [i for i, count in enumerate(counts) if count >= min_class_samples]
    if not keep_ids:
        raise ValueError(
            f"No classes meet --min-class-samples={min_class_samples}. "
            "Lower the threshold."
        )

    dropped = [moves[i] for i, count in enumerate(counts) if count < min_class_samples]
    kept = [moves[i] for i in keep_ids]

    keep_mask = np.isin(y_idx, np.array(keep_ids))
    x_filtered = x[keep_mask]
    y_idx_filtered_old = y_idx[keep_mask]

    old_to_new = {old: new for new, old in enumerate(keep_ids)}
    y_idx_filtered = np.array([old_to_new[int(old)] for old in y_idx_filtered_old], dtype=np.int32)
    y_filtered = np.eye(len(keep_ids), dtype=np.float32)[y_idx_filtered]

    kept_counts = np.bincount(y_idx_filtered, minlength=len(keep_ids)).astype(int)
    summary = {
        "min_class_samples": int(min_class_samples),
        "dropped_labels": dropped,
        "kept_labels": kept,
        "kept_counts": {kept[i]: int(kept_counts[i]) for i in range(len(kept))},
    }
    return x_filtered, y_filtered, kept, summary


def stratified_split_indices(
    y: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("--val-ratio must be in (0, 1).")

    rng = np.random.default_rng(seed)
    y_idx = np.argmax(y, axis=1)
    classes = np.unique(y_idx)
    train_idx: list[int] = []
    val_idx: list[int] = []

    for cls in classes:
        cls_indices = np.where(y_idx == cls)[0]
        rng.shuffle(cls_indices)

        if len(cls_indices) <= 1:
            train_idx.extend(cls_indices.tolist())
            continue

        val_count = int(round(len(cls_indices) * val_ratio))
        val_count = max(1, min(val_count, len(cls_indices) - 1))

        val_idx.extend(cls_indices[:val_count].tolist())
        train_idx.extend(cls_indices[val_count:].tolist())

    train_idx_arr = np.array(sorted(train_idx), dtype=np.int32)
    val_idx_arr = np.array(sorted(val_idx), dtype=np.int32)
    return train_idx_arr, val_idx_arr


def build_class_weights(y_train: np.ndarray) -> dict[int, float]:
    y_idx = np.argmax(y_train, axis=1)
    counts = np.bincount(y_idx, minlength=y_train.shape[1]).astype(np.float32)
    counts[counts == 0] = 1.0
    total = float(np.sum(counts))
    n_classes = float(len(counts))
    weights = total / (n_classes * counts)
    return {i: float(weights[i]) for i in range(len(weights))}


def main() -> None:
    args = parse_args()

    data_path = args.data_path
    model_path = "models"
    os.makedirs(model_path, exist_ok=True)

    # ----- Load data -----
    x = np.load(os.path.join(data_path, "X.npy"))
    y = np.load(os.path.join(data_path, "y.npy"))

    label_map_path = os.path.join(data_path, "label_map.json")
    if os.path.exists(label_map_path):
        with open(label_map_path, "r", encoding="utf-8") as f:
            moves = json.load(f)
    else:
        moves = [f"class_{i}" for i in range(y.shape[1])]

    initial_samples = x.shape[0]
    initial_label_count = y.shape[1]
    if args.min_class_samples > 0:
        x, y, moves, filter_summary = filter_underrepresented_classes(
            x=x,
            y=y,
            moves=moves,
            min_class_samples=args.min_class_samples,
        )
    else:
        filter_summary = {
            "min_class_samples": 0,
            "dropped_labels": [],
            "kept_labels": moves,
            "kept_counts": {},
        }

    seq_len = x.shape[1]
    feature_count = x.shape[2]
    label_count = y.shape[1]

    print("Data summary:")
    print("  Samples      :", x.shape[0], f"(from {initial_samples})")
    print("  Seq length   :", seq_len)
    print("  Feature count:", feature_count)
    print("  Num classes  :", label_count, f"(from {initial_label_count})")
    print("  Labels       :", moves)
    print("  Architecture :", args.arch)
    if filter_summary["dropped_labels"]:
        print("  Dropped      :", filter_summary["dropped_labels"])

    train_idx, val_idx = stratified_split_indices(
        y=y,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    print("Split summary:")
    print("  Train samples:", x_train.shape[0])
    print("  Val samples  :", x_val.shape[0])
    print("Optimization:")
    print("  Learning rate:", args.learning_rate)
    print("  Label smooth :", args.label_smoothing)

    model = build_model(
        arch=args.arch,
        seq_len=seq_len,
        feature_count=feature_count,
        label_count=label_count,
    )

    optimizer = Adam(learning_rate=args.learning_rate, clipnorm=1.0)
    loss = CategoricalCrossentropy(label_smoothing=args.label_smoothing)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    run_tag = args.run_tag.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(model_path, "runs", f"{args.arch}_{run_tag}")
    os.makedirs(run_dir, exist_ok=True)

    best_model_path = os.path.join(model_path, f"capoeira_{args.arch}_best.keras")
    final_model_path = os.path.join(model_path, f"capoeira_{args.arch}_final.keras")
    history_path = os.path.join(model_path, f"training_history_{args.arch}.json")
    labels_path = os.path.join(model_path, f"capoeira_{args.arch}_labels.json")
    train_meta_path = os.path.join(model_path, f"capoeira_{args.arch}_train_meta.json")

    run_best_model_path = os.path.join(run_dir, f"capoeira_{args.arch}_best.keras")
    run_final_model_path = os.path.join(run_dir, f"capoeira_{args.arch}_final.keras")
    run_history_path = os.path.join(run_dir, f"training_history_{args.arch}.json")
    run_labels_path = os.path.join(run_dir, f"capoeira_{args.arch}_labels.json")
    run_train_meta_path = os.path.join(run_dir, f"capoeira_{args.arch}_train_meta.json")

    print("Run metadata:")
    print("  Run tag      :", run_tag)
    print("  Run dir      :", run_dir)

    monitor_metric = "val_loss" if x_val.shape[0] > 0 else "loss"
    es = EarlyStopping(monitor=monitor_metric, patience=8, restore_best_weights=True)
    mc = ModelCheckpoint(run_best_model_path, save_best_only=True, monitor=monitor_metric)
    lr_scheduler = ReduceLROnPlateau(
        monitor=monitor_metric,
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    class_weight = None
    if not args.no_class_weights:
        class_weight = build_class_weights(y_train)
        print("Class weights:", class_weight)
    else:
        print("Class weights: disabled")

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val) if x_val.shape[0] > 0 else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        class_weight=class_weight,
        callbacks=[es, mc, lr_scheduler],
    )

    model.save(run_final_model_path)
    if os.path.exists(run_best_model_path):
        shutil.copy2(run_best_model_path, best_model_path)
    shutil.copy2(run_final_model_path, final_model_path)
    print(f"Saved final model to: {run_final_model_path}")
    print(f"Saved best model to : {run_best_model_path}")
    print(f"Promoted final model to: {final_model_path}")
    print(f"Promoted best model to : {best_model_path}")

    with open(run_history_path, "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)
    shutil.copy2(run_history_path, history_path)
    print(f"Saved training history to: {run_history_path}")
    print(f"Promoted training history to: {history_path}")

    with open(run_labels_path, "w", encoding="utf-8") as f:
        json.dump(moves, f, indent=2)
    shutil.copy2(run_labels_path, labels_path)
    print(f"Saved model labels to: {run_labels_path}")
    print(f"Promoted model labels to: {labels_path}")

    train_meta = {
        "arch": args.arch,
        "run_tag": run_tag,
        "run_dir": run_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "learning_rate": args.learning_rate,
        "label_smoothing": args.label_smoothing,
        "class_weights_enabled": not bool(args.no_class_weights),
        "initial_samples": int(initial_samples),
        "initial_label_count": int(initial_label_count),
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
        "final_label_count": int(label_count),
        "labels_path": labels_path,
        **filter_summary,
    }
    with open(run_train_meta_path, "w", encoding="utf-8") as f:
        json.dump(train_meta, f, indent=2)
    shutil.copy2(run_train_meta_path, train_meta_path)
    print(f"Saved training metadata to: {run_train_meta_path}")
    print(f"Promoted training metadata to: {train_meta_path}")


if __name__ == "__main__":
    main()
