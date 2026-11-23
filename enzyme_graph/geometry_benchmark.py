"""Nested evaluation of geometry-based graph classification."""

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .comparison import development_folds
from .data import load_split, read_graphs
from .graph_geometry import compute_distances, graph_descriptor
from .training import device_for, metrics, write_json


def distance_features(matrix, reference_labels):
    query_floor = np.maximum(np.min(matrix, axis=1), 1e-5)
    class_nearest = np.stack(
        [
            np.sort(matrix[:, reference_labels == label], axis=1)[:, :6]
            for label in range(6)
        ],
        axis=1,
    )
    nearest = class_nearest[:, :, 0]
    med = np.maximum(np.median(nearest, axis=1, keepdims=True), 1e-5)
    within = np.log((nearest / query_floor[:, None]).clip(1, 100))
    absolute = np.log(np.maximum(nearest, 1e-5))
    # Rank estimates how exceptional a close match is within this query graph.
    ordered = np.sort(matrix, axis=1)[:, :20]
    ranks = (ordered[:, :, None] < nearest[:, None]).sum(axis=1)
    median_votes = []
    for k in (5, 15):
        idx = np.argpartition(matrix, k, axis=1)[:, :k]
        median_votes.append(
            (reference_labels[idx][:, :, None] == np.arange(6)).sum(axis=1) / k
        )
    return np.stack(
        (
            absolute,
            within,
            np.log(med.clip(1e-5)).repeat(6, axis=1),
            np.log((nearest / med).clip(1e-3, 4)),
            np.log(np.maximum(class_nearest[:, :, 1], 1e-5)),
            np.log(np.maximum(class_nearest[:, :, 3], 1e-5)),
            ranks / 20,
            *median_votes,
        ),
        axis=-1,
    )


def _local_probabilities(table, distance, y, training, query, *, bandwidth, alpha):
    transform = StandardScaler().fit(table[training])
    X = transform.transform(table).clip(-5, 5)
    M = distance[np.ix_(query, training)]
    out = []
    for index, q in enumerate(query):
        d = M[index]
        pivot = max(np.quantile(d, 0.3), 1e-5)
        w = np.exp(-d / (pivot * bandwidth))
        w *= len(w) / w.sum()
        clf = LogisticRegression(C=alpha, max_iter=300, tol=1e-3)
        out.append(
            clf.fit(X[training], y[training], sample_weight=w).predict_proba(X[[q]])[0]
        )
    return np.stack(out)


def _kernel_probabilities(train, query, labels, regularization, estimators, index):
    if estimators is not None and index < len(estimators):
        estimator = estimators[index]
    else:
        estimator = SVC(
            C=regularization, kernel="precomputed", probability=True, random_state=42
        ).fit(train, labels)
        if estimators is not None:
            estimators.append(estimator)
    return estimator.predict_proba(query)


def _build_design(y, query, ref, distances, table, *, estimators=None):
    raw = []
    train = []
    valid = []
    queries = [d[np.ix_(query, ref)] for d in distances]
    for matrix, d in zip(distances, queries):
        a = matrix[np.ix_(ref, ref)]
        scale = np.maximum(np.median(a), 1e-6)
        train.append(a / scale)
        valid.append(d / scale)
        raw.append(distance_features(d, y[ref]))
    # Combine transformed distances within each view. Similarity weights are fixed.
    median_train = np.median(train, axis=0)
    median_valid = np.median(valid, axis=0)
    candidate = []
    for index, gamma in enumerate((0.6, 1.8, 3.6)):
        kt = np.mean(np.exp(-gamma * np.stack(train)), axis=0)
        kv = np.mean(np.exp(-gamma * np.stack(valid)), axis=0)
        candidate.append(_kernel_probabilities(kt, kv, y[ref], 4, estimators, index))
    normalize = StandardScaler().fit(table[ref])
    z = normalize.transform(table).clip(-5, 5)
    dtrain = euclidean_distances(z[ref], squared=True)
    dtest = euclidean_distances(z[query], z[ref], squared=True)
    r = max(np.median(dtrain), 1e-6)
    a = np.exp(-2 * dtrain / r)
    b = np.exp(-2 * dtest / r)
    for index, weight in enumerate((0.25, 0.65), 3):
        fit = weight * a + (1 - weight) * np.exp(-2 * median_train)
        test = weight * b + (1 - weight) * np.exp(-2 * median_valid)
        candidate.append(_kernel_probabilities(fit, test, y[ref], 4, estimators, index))
    # The nested reference labels can be used to fit a graph-local model.
    full = np.zeros((len(y), len(y)))
    full[np.ix_(query, ref)] = median_valid
    for alpha in (0.035, 0.15):
        candidate.append(
            _local_probabilities(
                table, full, y, ref, query, bandwidth=0.13, alpha=alpha
            )
        )
    means = np.stack([np.log(a.clip(1e-3, 1)) for a in candidate], axis=2)
    raw.append(means)
    raw.append(np.broadcast_to(np.eye(6), (len(query), 6, 6)))
    return np.concatenate(raw, axis=2)


def _adaptive_design(y, query, reference, distances, *, base, estimators=None):
    train = np.stack([d[np.ix_(reference, reference)] for d in distances])
    test = np.stack([d[np.ix_(query, reference)] for d in distances])
    median = np.maximum(np.median(train, axis=(1, 2)), 1e-5)
    train = np.median(train / median[:, None, None], axis=0)
    test = np.median(test / median[:, None, None], axis=0)
    neighbors = train.copy()
    np.fill_diagonal(neighbors, np.inf)
    features = []
    for index, k in enumerate((5, 15)):
        train_radius = np.maximum(
            np.partition(neighbors, k - 1, axis=1)[:, k - 1], 1e-3
        )
        test_radius = np.maximum(np.partition(test, k - 1, axis=1)[:, k - 1], 1e-3)
        local_train = train / np.sqrt(train_radius[:, None] * train_radius[None, :])
        local_test = test / np.sqrt(test_radius[:, None] * train_radius[None, :])
        features.append(distance_features(local_test, y[reference]))
        kernel = np.exp(-local_train) * 0.55 + np.exp(-2 * train) * 0.45
        query_kernel = np.exp(-local_test) * 0.55 + np.exp(-2 * test) * 0.45
        p = _kernel_probabilities(
            kernel, query_kernel, y[reference], 5, estimators, index
        )
        features.append(np.log(p.clip(1e-3, 1))[:, :, None])
    labels = y[reference]
    same = labels[:, None] == labels[None, :]
    for bandwidth in (0.23, 0.55):
        count = ((train < bandwidth) & same).sum(1)
        weights = np.exp(-test / bandwidth) / np.sqrt(np.maximum(count, 1))
        voters = weights @ np.eye(6)[labels]
        p = voters / voters.sum(1, keepdims=True)
        features.append(np.log(p.clip(1e-3, 1))[:, :, None])
    return np.concatenate((base, *features), axis=2)


OUTER_FOLDS = 16
OUTER_SEEDS = (731, 997)
INNER_FOLDS = 8
INNER_SEED = 97
META_REGULARIZATION = 0.12
_GEOMETRY = None


def _initialize_worker(
    labels: np.ndarray, descriptors: np.ndarray, distances: np.ndarray
) -> None:
    global _GEOMETRY
    _GEOMETRY = labels, descriptors, distances


def fit_calibrators(labels, descriptors, distances, train, *, seed):
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    training_features = [[], []]
    training_classes = []
    for reference_idx, calibration_idx in inner.split(
        descriptors[train], labels[train]
    ):
        reference, calibration = train[reference_idx], train[calibration_idx]
        training_classes.append(labels[calibration])
        basic = _build_design(labels, calibration, reference, distances, descriptors)
        extended = _adaptive_design(
            labels, calibration, reference, distances, base=basic
        )
        training_features[0].append(basic)
        training_features[1].append(extended)
    target = (np.concatenate(training_classes)[:, None] == np.arange(6)).ravel()
    calibrators = []
    for pieces in training_features:
        training = np.concatenate(pieces)
        estimator = make_pipeline(
            StandardScaler(), LogisticRegression(C=META_REGULARIZATION, max_iter=1_500)
        )
        calibrators.append(
            estimator.fit(training.reshape(-1, training.shape[-1]), target)
        )
    return calibrators


def calibrated_probabilities(estimator, features):
    logits = estimator.decision_function(features.reshape(-1, features.shape[-1]))
    return softmax(logits.reshape(-1, 6), axis=1)


def _score_partition(partition: tuple[int, int, np.ndarray, np.ndarray]) -> tuple:
    repeat, number, train, query = partition
    labels, descriptors, distances = _GEOMETRY
    with threadpool_limits(limits=1):
        calibrators = fit_calibrators(
            labels, descriptors, distances, train, seed=INNER_SEED + number
        )
        basic = _build_design(labels, query, train, distances, descriptors)
        extended = _adaptive_design(labels, query, train, distances, base=basic)
        scores = [
            calibrated_probabilities(estimator, features)
            for estimator, features in zip(calibrators, (basic, extended))
        ]
    return repeat, number, query, scores[0], scores[1]


def cached_distances(
    cache: Path, graphs: list, fingerprint: str, device: torch.device
) -> np.ndarray:
    ids = [graph.graph_id for graph in graphs]
    if cache.exists():
        with np.load(cache, allow_pickle=False) as archive:
            if (
                archive["sha256"].item() != fingerprint
                or archive["ids"].tolist() != ids
            ):
                raise ValueError(
                    f"{cache} belongs to a different graph dataset or split"
                )
            return archive["distances"]
    distances = compute_distances(graphs, device)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache.parent) as temporary:
        staging = Path(temporary) / cache.name
        np.savez_compressed(staging, distances=distances, sha256=fingerprint, ids=ids)
        staging.replace(cache)
    return distances


def run(
    train_csv: Path,
    split_path: Path,
    output: Path,
    cache: Path,
    device: torch.device,
    workers: int,
) -> dict:
    fingerprint = hashlib.sha256(train_csv.read_bytes()).hexdigest()
    source = read_graphs(train_csv, labeled=True)
    split = load_split(source, split_path)
    development = list(
        chain.from_iterable(development_folds(split["train"] + split["validation"]))
    )
    graphs = development + split["test"]
    labels = np.asarray([graph.label for graph in graphs])
    if min(np.bincount(labels, minlength=6)) < OUTER_FOLDS:
        raise ValueError(
            f"the {OUTER_FOLDS}-fold evaluation requires at least {OUTER_FOLDS} graphs per class"
        )
    descriptors = np.stack([graph_descriptor(graph) for graph in graphs])
    distances = cached_distances(cache, graphs, fingerprint, device)
    if (
        distances.shape != (10, len(graphs), len(graphs))
        or not np.isfinite(distances).all()
    ):
        raise ValueError(f"{cache} contains invalid distance channels")

    partitions = []
    for repeat, seed in enumerate(OUTER_SEEDS):
        outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed)
        partitions.extend(
            (repeat, number, train, query)
            for number, (train, query) in enumerate(outer.split(descriptors, labels), 1)
        )
    cut = len(development)
    partitions.insert(0, (-1, 0, np.arange(cut), np.arange(cut, len(graphs))))
    base = np.zeros((len(OUTER_SEEDS), len(graphs), 6))
    adaptive = np.zeros_like(base)
    fold_scores = []
    fixed_holdout = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(labels, descriptors, distances),
    ) as pool:
        for repeat, number, query, basic, adjusted in pool.map(
            _score_partition, partitions
        ):
            combined = (basic + adjusted) * 0.5
            summary = metrics(labels[query], combined.argmax(axis=1))
            if repeat == -1:
                fixed_holdout = summary
                print(
                    f"Original holdout macro-F1 {summary['macro_f1']:.6f}", flush=True
                )
            else:
                base[repeat, query] = basic
                adaptive[repeat, query] = adjusted
                fold_scores.append({"repeat": repeat + 1, "fold": number, **summary})
                print(
                    f"Repeat {repeat + 1} fold {number}: macro-F1 {summary['macro_f1']:.6f}",
                    flush=True,
                )
    probabilities = (base.mean(axis=0) + adaptive.mean(axis=0)) * 0.5
    result = {
        "outer_folds": OUTER_FOLDS,
        "outer_seeds": list(OUTER_SEEDS),
        "inner_folds": INNER_FOLDS,
        "graphs_evaluated": len(graphs),
        "train_graphs_per_outer_fold": len(graphs) - len(partitions[1][3]),
        "data_sha256": fingerprint,
        "out_of_fold": metrics(labels, probabilities.argmax(axis=1)),
        "base_out_of_fold": metrics(labels, base.mean(axis=0).argmax(axis=1)),
        "adaptive_out_of_fold": metrics(labels, adaptive.mean(axis=0).argmax(axis=1)),
        "original_holdout": fixed_holdout,
        "fold_scores": fold_scores,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "geometry.json", result)
    np.savez_compressed(
        output / "probabilities.npz",
        probabilities=probabilities,
        base=base,
        adaptive=adaptive,
        graph_ids=[graph.graph_id for graph in graphs],
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--split", type=Path, default=Path("runs/split.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/geometry"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    torch.set_num_threads(3)
    result = run(
        args.data,
        args.split,
        args.output,
        args.output / "distances.npz",
        device_for(args.device),
        args.workers,
    )
    print(f"Nested out-of-fold macro-F1: {result['out_of_fold']['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
