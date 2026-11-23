"""Select residual message-passing models using development folds."""

import random
from pathlib import Path
from statistics import mean, median, stdev

import torch

from .benchmark import SEEDS, plot_results
from .data import CLASS_COUNT, Graph
from .models import MESSAGE_PASSING
from .training import evaluate, train, write_json

FOLD_COUNT = 5
FOLD_SEED = 93
LEARNING_RATE = 0.001


def development_folds(graphs: list[Graph]) -> list[list[Graph]]:
    rng = random.Random(FOLD_SEED)
    folds = [[] for _ in range(FOLD_COUNT)]
    for label in range(CLASS_COUNT):
        group = [graph for graph in graphs if graph.label == label]
        if len(group) < FOLD_COUNT:
            raise ValueError("development folds require at least five examples per class")
        rng.shuffle(group)
        for index, graph in enumerate(group):
            folds[index % FOLD_COUNT].append(graph)
    return folds


def run_comparison(
    split: dict[str, list[Graph]], output: Path, *,
    epochs: int, patience: int, batch_size: int, device: torch.device,
) -> dict:
    development = split["train"] + split["validation"]
    folds = development_folds(development)
    candidates = {}
    for architecture in MESSAGE_PASSING:
        results = []
        for index, validation in enumerate(folds):
            training = [graph for i, fold in enumerate(folds) if i != index for graph in fold]
            checkpoint = output / f"{architecture}-fold-{index + 1}.pt"
            run = train(
                training, validation, checkpoint, architecture=architecture,
                lr=LEARNING_RATE, seed=SEEDS[0], epochs=epochs, patience=patience,
                batch_size=batch_size, device=device,
            )
            results.append({key: value for key, value in run.items() if key != "history"})
            print(f"{architecture} fold={index + 1} best_epoch={run['best_epoch']} "
                  f"validation_f1={run['validation_f1']:.4f}", flush=True)
        candidates[architecture] = {
            "folds": results,
            "mean_validation_f1": mean(run["validation_f1"] for run in results),
            "refit_epochs": int(median(run["best_epoch"] for run in results)),
        }
    winner = max(candidates, key=lambda name: candidates[name]["mean_validation_f1"])
    print(f"Selected {winner} by development macro-F1", flush=True)

    # Freeze the residual candidate and its epoch count before scoring the original test split.
    comparisons = {}
    winner_auc = None
    for architecture, lr, refit_epochs in (
        ("gin", 0.003, min(36, epochs)),
        (winner, LEARNING_RATE, candidates[winner]["refit_epochs"]),
    ):
        results = []
        for seed in SEEDS:
            checkpoint = output / f"{architecture}-refit-s{seed}.pt"
            run = train(
                development, None, checkpoint, architecture=architecture,
                lr=lr, seed=seed, epochs=refit_epochs, patience=patience,
                batch_size=batch_size, device=device,
            )
            scores, probability, labels = evaluate(split["test"], checkpoint, device, batch_size)
            result = {key: value for key, value in run.items() if key != "history"}
            result.update(test=scores, checkpoint=checkpoint.name)
            results.append(result)
            if architecture == winner and seed == SEEDS[0]:
                winner_auc = plot_results(output, winner, scores, probability, labels)
            print(f"{architecture} seed={seed} test_f1={scores['macro_f1']:.4f} "
                  f"test_accuracy={scores['accuracy']:.4f}", flush=True)
        comparisons[architecture] = {"runs": results}
        for metric in ("macro_f1", "accuracy"):
            values = [run["test"][metric] for run in results]
            comparisons[architecture][f"mean_test_{metric}"] = mean(values)
            comparisons[architecture][f"std_test_{metric}"] = stdev(values)
    summary = {
        "winner": winner,
        "fold_seed": FOLD_SEED,
        "fold_sizes": list(map(len, folds)),
        "development_count": len(development),
        "test_count": len(split["test"]),
        "candidates": candidates,
        "comparisons": comparisons,
        "winner_roc_auc": winner_auc,
    }
    write_json(output / "comparison.json", summary)
    return summary
