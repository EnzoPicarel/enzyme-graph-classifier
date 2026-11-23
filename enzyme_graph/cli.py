"""Command-line entry points for the ENZYMES benchmark."""

import argparse
import json
from pathlib import Path

import torch

from .benchmark import run_benchmark
from .data import describe, load_split, read_graphs
from .models import ARCHITECTURES
from .training import device_for, evaluate, test_model, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect")

    def with_split(command):
        command.add_argument("--split", type=Path, default=Path("runs/split.json"))
        return command

    def training_options(command):
        command.add_argument("--epochs", type=int, default=100)
        command.add_argument("--patience", type=int, default=15)

    training = with_split(commands.add_parser("train"))
    training_options(training)
    training.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    training.add_argument("--checkpoint", type=Path, required=True)
    training.add_argument("--lr", type=float, default=0.001)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--all-labeled", action="store_true", help="Train for the exact epoch count without validation")

    evaluation = with_split(commands.add_parser("evaluate"))
    evaluation.add_argument("--checkpoint", type=Path, required=True)

    prediction = commands.add_parser("predict")
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)

    benchmark = with_split(commands.add_parser("benchmark"))
    training_options(benchmark)
    benchmark.add_argument("--output", type=Path, default=Path("runs/benchmark"))

    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error("--threads and --batch-size must be positive")
    torch.set_num_threads(args.threads)

    try:
        run(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")


def run(args: argparse.Namespace) -> None:
    if args.command == "inspect":
        print(json.dumps({
            "labeled": describe(read_graphs(args.data / "train.csv", labeled=True)),
            "unlabeled": describe(read_graphs(args.data / "test.csv", labeled=False)),
        }, indent=2))
        return

    device = device_for(args.device)
    if args.command == "predict":
        test_model(args.checkpoint, args.data / "test.csv", args.data / "sample_submission.csv",
                   args.output, device=device, batch_size=args.batch_size)
        print(f"Saved predictions to {args.output}")
        return

    graphs = read_graphs(args.data / "train.csv", labeled=True)
    all_labeled = args.command == "train" and args.all_labeled
    split = None if all_labeled else load_split(graphs, args.split)
    if args.command == "train":
        train(
            graphs if all_labeled else split["train"], None if all_labeled else split["validation"],
            args.checkpoint, architecture=args.architecture, lr=args.lr, seed=args.seed,
            epochs=args.epochs, patience=args.patience, batch_size=args.batch_size, device=device, verbose=True,
        )
        print(f"Saved checkpoint to {args.checkpoint}")
    elif args.command == "evaluate":
        scores, _, _ = evaluate(split["test"], args.checkpoint, device, args.batch_size)
        print(json.dumps(scores, indent=2))
    else:
        result = run_benchmark(
            split, args.output, epochs=args.epochs, patience=args.patience, batch_size=args.batch_size, device=device,
        )
        print(f"Validation winner: {result['winner']}. Results saved to {args.output}")


if __name__ == "__main__":
    main()
