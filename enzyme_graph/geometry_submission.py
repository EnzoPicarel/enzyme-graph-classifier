"""Save a graph-similarity ensemble and reload it for CSV prediction."""

import argparse
import csv
from itertools import chain
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .comparison import development_folds
from .data import load_split, read_graphs
from .geometry_benchmark import (
    INNER_SEED,
    _adaptive_design,
    _build_design,
    calibrated_probabilities,
    fit_calibrators,
)
from .graph_geometry import compute_distances, graph_descriptor
from .training import device_for


def train_model(
    train_csv: Path,
    split_path: Path,
    model_path: Path,
    *,
    device: torch.device,
    geometry_test: Path | None = None,
) -> None:
    """Fit the reference estimators and both calibrators using all labeled graphs."""
    split = load_split(read_graphs(train_csv, labeled=True), split_path)
    references = list(
        chain.from_iterable(development_folds(split["train"] + split["validation"]))
    )
    references.extend(split["test"])
    # The competition release was featurized in one batch. Preserve its block
    # layout to reproduce the reference distances at floating-point precision.
    context = read_graphs(geometry_test, labeled=False) if geometry_test else []
    graphs = references + context
    labels = np.array([graph.label for graph in graphs])
    descriptors = np.stack([graph_descriptor(graph) for graph in graphs])
    distances = compute_distances(graphs, device)
    train = np.arange(len(references))
    base_estimators = []
    adaptive_estimators = []
    with threadpool_limits(limits=1):
        calibrators = fit_calibrators(
            labels, descriptors, distances, train, seed=INNER_SEED
        )
        # A reference row supplies the query shape while the kernels fit on all reference rows.
        base = _build_design(
            labels, train[:1], train, distances, descriptors, estimators=base_estimators
        )
        _adaptive_design(
            labels,
            train[:1],
            train,
            distances,
            base=base,
            estimators=adaptive_estimators,
        )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "version": 1,
            "references": references,
            "descriptors": descriptors[: len(train)],
            "distances": distances[:, : len(train), : len(train)],
            "base_estimators": base_estimators,
            "adaptive_estimators": adaptive_estimators,
            "calibrators": calibrators,
        },
        model_path,
        compress=3,
    )


def test_model(
    model_path: Path,
    test_csv: Path,
    sample_csv: Path,
    output: Path,
    *,
    device: torch.device,
) -> None:
    """Load a trusted model artifact and produce one prediction per test graph."""
    state = joblib.load(model_path)
    if state["version"] != 1:
        raise ValueError("unsupported geometry model version")
    references = state["references"]
    query_graphs = read_graphs(test_csv, labeled=False)
    graphs = references + query_graphs
    train = np.arange(len(references))
    query = np.arange(len(references), len(graphs))
    labels = np.array([graph.label for graph in graphs])
    descriptors = np.vstack(
        (state["descriptors"], [graph_descriptor(graph) for graph in query_graphs])
    )
    distances = compute_distances(graphs, device)
    distances[:, : len(train), : len(train)] = state["distances"]
    with threadpool_limits(limits=1):
        base = _build_design(
            labels,
            query,
            train,
            distances,
            descriptors,
            estimators=state["base_estimators"],
        )
        extended = _adaptive_design(
            labels,
            query,
            train,
            distances,
            base=base,
            estimators=state["adaptive_estimators"],
        )
        probabilities = [
            calibrated_probabilities(estimator, features)
            for estimator, features in zip(state["calibrators"], (base, extended))
        ]
    labels_out = ((probabilities[0] + probabilities[1]) * 0.5).argmax(axis=1) + 1
    predicted = dict(zip((graph.graph_id for graph in query_graphs), labels_out))
    with sample_csv.open(newline="", encoding="utf-8-sig") as file:
        ids = [row["graph_id"] for row in csv.DictReader(file)]
    if len(ids) != len(predicted) or set(ids) != predicted.keys():
        raise ValueError("sample submission must list every test ID exactly once")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["graph_id", "label"])
        writer.writerows((graph_id, predicted[graph_id]) for graph_id in ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit")
    fit.add_argument("--train", type=Path, required=True)
    fit.add_argument("--split", type=Path, required=True)
    fit.add_argument("--model", type=Path, required=True)
    fit.add_argument("--geometry-test", type=Path)
    predict = commands.add_parser("predict")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--test", type=Path, required=True)
    predict.add_argument("--sample", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(3)
    device = device_for(args.device)
    if args.command == "fit":
        train_model(
            args.train,
            args.split,
            args.model,
            device=device,
            geometry_test=args.geometry_test,
        )
        print(f"Saved ensemble to {args.model}")
    else:
        test_model(args.model, args.test, args.sample, args.output, device=device)
        print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
