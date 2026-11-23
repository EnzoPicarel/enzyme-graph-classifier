"""Train, score, and reload graph classifiers."""

import csv
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import CLASS_COUNT, Graph, collate, fit_scaler, prepare, read_graphs
from .models import GraphClassifier


def device_for(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("PyTorch cannot access a GPU")
    return torch.device(name)


def loader(graphs: list[Graph], scaler: tuple, batch_size: int, seed: int | None = None) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(
        prepare(graphs, scaler), batch_size=batch_size, shuffle=seed is not None,
        generator=generator, collate_fn=collate,
    )


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=int)
    np.add.at(confusion, (labels, predictions), 1)
    diagonal = confusion.diagonal()
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros(CLASS_COUNT), where=denominator != 0)
    return {
        "macro_f1": float(f1.mean()),
        "accuracy": float(diagonal.sum() / len(labels)),
        "class_f1": f1.tolist(),
        "confusion": confusion.tolist(),
    }


@torch.inference_mode()
def infer(model: GraphClassifier, batches: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    labels = []
    for features, adjacency, mask, target in batches:
        logits = model(features.to(device), adjacency.to(device), mask.to(device))
        probabilities.append(logits.softmax(dim=1).cpu())
        labels.append(target)
    return torch.cat(probabilities).numpy(), torch.cat(labels).numpy()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def train(
    graphs: list[Graph], validation: list[Graph] | None, checkpoint: Path, *,
    architecture: str, lr: float, seed: int, epochs: int, patience: int,
    batch_size: int, device: torch.device, verbose: bool = False,
) -> dict:
    if min(lr, epochs, patience, batch_size) <= 0:
        raise ValueError("learning rate, epochs, patience, and batch size must be positive")
    random.seed(seed)
    torch.manual_seed(seed)
    scaler = fit_scaler(graphs)
    batches = loader(graphs, scaler, batch_size, seed)
    validation_batches = loader(validation, scaler, batch_size) if validation else None
    model = GraphClassifier(architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    state = None
    best_f1 = -1.0
    best_epoch = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = torch.zeros((), device=device)
        for features, adjacency, mask, labels in batches:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features.to(device), adjacency.to(device), mask.to(device))
            loss = F.cross_entropy(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(labels)
        epoch_loss = float(loss_sum / len(graphs))
        if not np.isfinite(epoch_loss):
            raise ValueError(f"non-finite training loss at epoch {epoch}")
        record = {"epoch": epoch, "train_loss": epoch_loss}
        history.append(record)
        if validation_batches is not None:
            probability, labels = infer(model, validation_batches, device)
            scores = metrics(labels, probability.argmax(axis=1))
            record.update(validation_f1=scores["macro_f1"], validation_accuracy=scores["accuracy"])
            if scores["macro_f1"] > best_f1:
                best_f1 = scores["macro_f1"]
                best_epoch = epoch
                state = {name: tensor.detach().to("cpu", copy=True) for name, tensor in model.state_dict().items()}
        if verbose:
            print(json.dumps(record), flush=True)
        if validation_batches is not None and epoch - best_epoch >= patience:
            break
    if validation_batches is None:
        best_epoch = len(history)
        state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    elapsed = time.perf_counter() - started
    result = {
        "architecture": architecture, "lr": lr, "seed": seed, "best_epoch": best_epoch,
        "epochs_run": len(history), "batch_size": batch_size,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "seconds": elapsed, "device": device.type,
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
        "torch": str(torch.__version__), "threads": torch.get_num_threads(), "history": history,
    }
    if validation_batches is not None:
        result["validation_f1"] = best_f1
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"architecture": architecture, "state": state, "scaler": scaler}, checkpoint)
    write_json(checkpoint.with_suffix(".json"), result)
    return result


def load_model(checkpoint: Path, device: torch.device) -> tuple[GraphClassifier, tuple]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = GraphClassifier(saved["architecture"]).to(device)
    model.load_state_dict(saved["state"])
    return model, saved["scaler"]


def evaluate(graphs: list[Graph], checkpoint: Path, device: torch.device, batch_size: int) -> tuple[dict, np.ndarray, np.ndarray]:
    model, scaler = load_model(checkpoint, device)
    probability, labels = infer(model, loader(graphs, scaler, batch_size), device)
    return metrics(labels, probability.argmax(axis=1)), probability, labels


def test_model(
    checkpoint: Path, test_csv: Path, sample_csv: Path, output: Path,
    *, device: torch.device, batch_size: int = 32,
) -> None:
    graphs = read_graphs(test_csv, labeled=False)
    model, scaler = load_model(checkpoint, device)
    probability, _ = infer(model, loader(graphs, scaler, batch_size), device)
    predicted = {graph.graph_id: int(index) + 1 for graph, index in zip(graphs, probability.argmax(axis=1))}
    with sample_csv.open(newline="", encoding="utf-8-sig") as file:
        ids = [row["graph_id"] for row in csv.DictReader(file)]
    if len(ids) != len(predicted) or set(ids) != predicted.keys():
        raise ValueError("sample submission must list every test ID exactly once")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["graph_id", "label"])
        writer.writerows((graph_id, predicted[graph_id]) for graph_id in ids)
