import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import torch
from test_pipeline import write_fixture

from enzyme_graph.data import read_graphs
from enzyme_graph.tu_data import convert_tu


class PublicWorkflowTest(unittest.TestCase):
    def test_tu_archive_conversion_remaps_global_edge_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "ENZYMES.zip"
            with ZipFile(archive_path, "w") as archive:
                for name, content in {
                    "graph_labels": "2\n6\n",
                    "graph_indicator": "1\n2\n1\n2\n",
                    "node_labels": "1\n2\n3\n1\n",
                    "node_attributes": "\n".join(
                        ",".join([str(row)] * 18) for row in range(4)
                    ),
                    "A": "1,3\n3,1\n2,4\n4,2\n",
                }.items():
                    archive.writestr(f"ENZYMES/ENZYMES_{name}.txt", content)

            output = root / "public" / "train.csv"
            self.assertEqual(convert_tu(archive_path, output), 2)
            graphs = read_graphs(output, labeled=True)
            self.assertEqual([graph.graph_id for graph in graphs], ["tu-1", "tu-2"])
            self.assertEqual([graph.label for graph in graphs], [1, 5])
            torch.testing.assert_close(
                graphs[0].features[:, 0], torch.tensor([0.0, 2.0])
            )
            torch.testing.assert_close(
                graphs[1].adjacency, torch.tensor([[0.0, 1.0], [1.0, 0.0]])
            )

    @unittest.skipUnless(
        all(
            importlib.util.find_spec(module) is not None
            for module in ("scipy", "sklearn", "joblib")
        ),
        "geometry dependencies are optional",
    )
    def test_saved_geometry_model_reloads_and_honors_sample_order(self):
        from enzyme_graph.geometry_submission import test_model, train_model

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            model = root / "model.joblib"
            device = torch.device("cpu")
            train_model(root / "train.csv", root / "split.json", model, device=device)
            output = root / "predictions.csv"
            test_model(
                model,
                root / "test.csv",
                root / "sample_submission.csv",
                output,
                device=device,
            )
            with output.open(newline="") as file:
                reader = csv.DictReader(file)
                self.assertEqual(reader.fieldnames, ["graph_id", "label"])
                predictions = list(reader)
            self.assertEqual(
                [row["graph_id"] for row in predictions],
                [f"{label}-8" for label in range(6, 0, -1)],
            )
            self.assertTrue(all(1 <= int(row["label"]) <= 6 for row in predictions))
