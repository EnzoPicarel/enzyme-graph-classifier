import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from enzyme_graph.benchmark import roc_curve
from enzyme_graph.cli import main


def write_fixture(directory: Path) -> None:
    fields = ["graph_id", "num_nodes", "node_features", "node_labels", "edge_index"]
    train_rows = []
    test_rows = []
    for label in range(1, 7):
        for offset in range(9):
            nodes = 2 + (offset % 2)
            edge_index = [[0, 1], [1, 0]] if offset else [[], []]
            row = [
                f"{label}-{offset}", nodes, json.dumps([[label + offset / 10] * 18] * nodes),
                json.dumps([1 + label % 3] * nodes), json.dumps(edge_index),
            ]
            if offset < 8:
                train_rows.append([*row, label])
            else:
                test_rows.append(row)
    for name, columns, rows in (
        ("train.csv", fields + ["label"], train_rows),
        ("test.csv", fields, test_rows),
        ("sample_submission.csv", ["graph_id", "label"], [[row[0], 1] for row in reversed(test_rows)]),
    ):
        with (directory / name).open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            writer.writerows(rows)


class PipelineTest(unittest.TestCase):
    def test_cpu_cli_and_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            split = root / "split.json"
            checkpoint = root / "model.pt"
            output = root / "submission.csv"

            def run(*args):
                captured = io.StringIO()
                with patch.object(sys, "argv", ["enzyme-graph", "--data", str(root), "--device", "cpu", *map(str, args)]):
                    with redirect_stdout(captured):
                        main()
                return captured.getvalue()

            profile = json.loads(run("inspect"))
            self.assertEqual(profile["labeled"]["graphs"], 48)
            self.assertEqual(profile["unlabeled"]["graphs"], 6)
            run("train", "--architecture", "gcn2", "--epochs", 2, "--checkpoint", checkpoint, "--split", split)
            groups = json.loads(split.read_text())
            self.assertEqual({name: len(ids) for name, ids in groups.items()}, {"train": 36, "validation": 6, "test": 6})
            self.assertEqual(len(set(sum(groups.values(), []))), 48)
            scores = json.loads(run("evaluate", "--checkpoint", checkpoint, "--split", split))
            self.assertEqual(sum(map(sum, scores["confusion"])), 6)
            run("predict", "--checkpoint", checkpoint, "--output", output)
            with output.open(newline="") as file:
                reader = csv.DictReader(file)
                self.assertEqual(reader.fieldnames, ["graph_id", "label"])
                rows = list(reader)
            self.assertEqual([row["graph_id"] for row in rows], [f"{label}-8" for label in range(6, 0, -1)])
            self.assertTrue(all(1 <= int(row["label"]) <= 6 for row in rows))

    def test_roc_treats_probability_ties_as_one_threshold(self):
        labels = np.array([True, False, True, False])
        _, _, area = roc_curve(labels, np.array([0.9, 0.9, 0.4, 0.1]))
        self.assertAlmostEqual(area, 0.625)
