"""Run the controlled model comparison and plot held-out performance."""

from pathlib import Path

import matplotlib
import numpy as np

from .data import CLASS_COUNT, Graph
from .models import ARCHITECTURES
from .training import evaluate, metrics, train, write_json

matplotlib.use("Agg")
from matplotlib import pyplot as plt

LEARNING_RATES = (0.0003, 0.001, 0.003)
SEEDS = (42, 43, 44)


def roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-scores, kind="stable")
    labels, scores = labels[order], scores[order]
    threshold_ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    positives = labels.sum()
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("ROC requires positive and negative examples")
    true_positives = np.cumsum(labels)[threshold_ends]
    tpr = np.r_[0, true_positives / positives]
    fpr = np.r_[0, (threshold_ends + 1 - true_positives) / negatives]
    widths = np.diff(fpr)
    return fpr, tpr, float(np.sum(widths * (tpr[1:] + tpr[:-1]) / 2))


def plot_results(directory: Path, architecture: str, scores: dict, probability: np.ndarray, labels: np.ndarray) -> dict:
    fig, ax = plt.subplots(figsize=(7, 5.5), layout="constrained")
    auc = {}
    for label in range(CLASS_COUNT):
        fpr, tpr, value = roc_curve(labels == label, probability[:, label])
        auc[str(label + 1)] = value
        ax.plot(fpr, tpr, label=f"Class {label + 1} · {value:.3f}")
    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02),
           title=f"{architecture.upper()} · held-out ROC by class")
    ax.legend(loc="lower right")
    fig.savefig(directory / "roc.png", dpi=160)
    plt.close(fig)

    confusion = np.array(scores["confusion"])
    fig, ax = plt.subplots(figsize=(6, 5.5), layout="constrained")
    image = ax.imshow(confusion, cmap="Blues", vmin=0)
    for row, column in np.ndindex(confusion.shape):
        count = confusion[row, column]
        ax.text(column, row, str(count), ha="center", va="center",
                color="white" if count > confusion.max() / 2 else "black")
    ax.set(xlabel="Predicted class", ylabel="Actual class", title=f"{architecture.upper()} · held-out confusion matrix",
           xticks=range(CLASS_COUNT), yticks=range(CLASS_COUNT),
           xticklabels=range(1, CLASS_COUNT + 1), yticklabels=range(1, CLASS_COUNT + 1))
    fig.colorbar(image, ax=ax, label="Graphs")
    fig.savefig(directory / "confusion.png", dpi=160)
    plt.close(fig)
    return auc


def run_benchmark(split: dict[str, list[Graph]], directory: Path, *, epochs: int, patience: int, batch_size: int, device) -> dict:
    def run(architecture: str, lr: float, seed: int) -> dict:
        checkpoint = directory / f"{architecture}-{lr:g}-s{seed}.pt"
        result = train(
            split["train"], split["validation"], checkpoint, architecture=architecture,
            lr=lr, seed=seed, epochs=epochs, patience=patience, batch_size=batch_size, device=device,
        )
        result = {key: value for key, value in result.items() if key != "history"}
        result["checkpoint"] = checkpoint.name
        print(f"{architecture} lr={lr:g} seed={seed} epoch={result['best_epoch']} "
              f"validation_f1={result['validation_f1']:.4f} time={result['seconds']:.2f}s", flush=True)
        return result

    models = {}
    for architecture in ARCHITECTURES:
        trials = [run(architecture, lr, SEEDS[0]) for lr in LEARNING_RATES]
        primary = max(trials, key=lambda trial: trial["validation_f1"])
        selected = [primary] + [run(architecture, primary["lr"], seed) for seed in SEEDS[1:]]
        models[architecture] = {
            "trials": trials, "selected": selected,
            "mean_validation_f1": float(np.mean([record["validation_f1"] for record in selected])),
        }
    # Freeze the architecture and learning rates before opening the held-out split.
    winner = max(models, key=lambda name: models[name]["mean_validation_f1"])
    winner_auc = None
    for architecture, group in models.items():
        for run_record in group["selected"]:
            score, probability, labels = evaluate(
                split["test"], directory / run_record["checkpoint"], device, batch_size,
            )
            run_record["test"] = score
            if architecture == winner and run_record["seed"] == SEEDS[0]:
                winner_auc = plot_results(directory, winner, score, probability, labels)
        for name in ("macro_f1", "accuracy"):
            values = [record["test"][name] for record in group["selected"]]
            group[f"mean_test_{name}"] = float(np.mean(values))
            group[f"std_test_{name}"] = float(np.std(values, ddof=1))
        group["mean_training_seconds"] = float(np.mean([record["seconds"] for record in group["selected"]]))
        print(f"{architecture} test_f1={group['mean_test_macro_f1']:.4f} "
              f"test_accuracy={group['mean_test_accuracy']:.4f}", flush=True)
    train_labels = [graph.label for graph in split["train"]]
    test_labels = np.array([graph.label for graph in split["test"]])
    majority_label = int(np.bincount(train_labels, minlength=CLASS_COUNT).argmax())
    summary = {
        "winner": winner, "models": models, "winner_roc_auc": winner_auc,
        "majority": metrics(test_labels, np.full_like(test_labels, majority_label)),
        "split_counts": {name: len(graphs) for name, graphs in split.items()},
    }
    write_json(directory / "benchmark.json", summary)
    return summary
