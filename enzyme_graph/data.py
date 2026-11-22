"""Read ENZYMES graphs, persist the benchmark split, and build padded batches."""

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import pad_sequence

CLASS_COUNT = 6
FEATURE_COUNT = 18
NODE_TYPE_COUNT = 3


@dataclass
class Graph:
    graph_id: str
    features: torch.Tensor
    node_types: torch.Tensor
    adjacency: torch.Tensor
    label: int


def read_graphs(path: Path, *, labeled: bool) -> list[Graph]:
    graphs = []
    seen = set()
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"graph_id", "num_nodes", "node_features", "node_labels", "edge_index"}
        if labeled:
            required.add("label")
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing columns {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            graph_id = row["graph_id"]
            if not graph_id or graph_id in seen:
                raise ValueError(f"{path}: empty or duplicate graph ID {graph_id!r}")
            seen.add(graph_id)
            try:
                count = int(row["num_nodes"])
                features = torch.tensor(json.loads(row["node_features"]), dtype=torch.float32)
                node_types = torch.tensor(json.loads(row["node_labels"]))
                edges = torch.tensor(json.loads(row["edge_index"]))
                label = int(row["label"]) - 1 if labeled else -1
                if count < 1 or features.shape != (count, FEATURE_COUNT):
                    raise ValueError("incorrect node feature shape")
                if not torch.isfinite(features).all():
                    raise ValueError("non-finite node features")
                if (node_types.shape != (count,) or node_types.dtype != torch.long
                        or not ((node_types >= 1) & (node_types <= NODE_TYPE_COUNT)).all()):
                    raise ValueError("invalid node labels")
                if edges.ndim != 2 or edges.shape[0] != 2:
                    raise ValueError("expected an edge index of shape (2, E)")
                if edges.numel() and (edges.dtype != torch.long
                                     or not ((edges >= 0) & (edges < count)).all()):
                    raise ValueError("invalid edge indices")
                if labeled and not 0 <= label < CLASS_COUNT:
                    raise ValueError("invalid graph label")
                adjacency = torch.zeros(count, count)
                # The observed data are undirected and use zero-based node indices.
                source, target = edges.long()
                adjacency[target, source] = 1
                if not torch.equal(adjacency, adjacency.T) or adjacency.diagonal().any():
                    raise ValueError("expected undirected edges without self-loops")
                graphs.append(Graph(graph_id, features, node_types - 1, adjacency, label))
            except (ValueError, TypeError, RuntimeError, OverflowError) as error:
                raise ValueError(f"{path}: graph {graph_id}: {error}") from error
    if not graphs:
        raise ValueError(f"{path}: no graphs")
    return graphs


def describe(graphs: list[Graph]) -> dict:
    labels = [graph.label for graph in graphs]
    sizes = [len(graph.features) for graph in graphs]
    return {
        "graphs": len(graphs),
        "nodes": sum(sizes),
        "node_range": [min(sizes), max(sizes)],
        "undirected_edges": sum(int(graph.adjacency.sum()) for graph in graphs) // 2,
        "numeric_features": FEATURE_COUNT,
        "node_categories": {
            str(category + 1): sum(int((g.node_types == category).sum()) for g in graphs)
            for category in range(NODE_TYPE_COUNT)
        },
        "graph_classes": (
            {str(label + 1): labels.count(label) for label in range(CLASS_COUNT)}
            if labels[0] >= 0 else {}
        ),
        "non_finite_node_features": 0,
    }


def load_split(graphs: list[Graph], path: Path) -> dict[str, list[Graph]]:
    if not path.exists():
        rng = random.Random(42)
        ids = {"train": [], "validation": [], "test": []}
        for label in range(CLASS_COUNT):
            group = sorted(graph.graph_id for graph in graphs if graph.label == label)
            holdout = len(group) // 8
            if not holdout:
                raise ValueError("stratification requires at least eight graphs per class")
            rng.shuffle(group)
            ids["test"].extend(group[:holdout])
            ids["validation"].extend(group[holdout:2 * holdout])
            ids["train"].extend(group[2 * holdout:])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as file:
            json.dump(ids, file, indent=2)
            file.write("\n")
    ids = json.loads(path.read_text(encoding="utf-8"))
    by_id = {graph.graph_id: graph for graph in graphs}
    assigned = ids["train"] + ids["validation"] + ids["test"]
    if len(assigned) != len(by_id) or set(assigned) != set(by_id):
        raise ValueError(f"{path}: split must assign each labeled graph exactly once")
    return {name: [by_id[graph_id] for graph_id in ids[name]] for name in ("train", "validation", "test")}


def fit_scaler(graphs: list[Graph]) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = torch.cat([graph.features for graph in graphs])
    return matrix.mean(dim=0), matrix.std(dim=0, correction=0).clamp_min(1e-6)


def prepare(graphs: list[Graph], scaler: tuple[torch.Tensor, torch.Tensor]) -> list[tuple]:
    mean, std = scaler
    return [
        (torch.cat(((graph.features - mean) / std, one_hot(graph.node_types, NODE_TYPE_COUNT)), dim=1),
         graph.adjacency, graph.label)
        for graph in graphs
    ]


def collate(items: list[tuple]) -> tuple[torch.Tensor, ...]:
    features, edges, labels = zip(*items)
    padded = pad_sequence(features, batch_first=True)
    batch, nodes, _ = padded.shape
    mask = torch.arange(nodes)[None, :] < torch.tensor([len(x) for x in features])[:, None]
    adjacency = padded.new_zeros(batch, nodes, nodes)
    for index, matrix in enumerate(edges):
        adjacency[index, :len(matrix), :len(matrix)] = matrix
    return padded, adjacency, mask, torch.tensor(labels)
