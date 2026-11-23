"""Convert the public TUDataset ENZYMES archive to the benchmark CSV schema."""

import csv
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from .data import read_graphs

_FIELDS = (
    "graph_id",
    "num_nodes",
    "node_features",
    "node_labels",
    "edge_index",
    "label",
)


def _lines(archive: ZipFile, suffix: str) -> list[str]:
    filename = f"ENZYMES_{suffix}.txt"
    matches = [
        name for name in archive.namelist() if PurePosixPath(name).name == filename
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {filename} in the TUDataset archive")
    return archive.read(matches[0]).decode("utf-8-sig").splitlines()


def convert_tu(archive_path: Path, output: Path) -> int:
    """Write the labeled graphs with graph-local node indices and stable TU IDs."""
    with ZipFile(archive_path) as archive:
        graph_labels = [int(line) for line in _lines(archive, "graph_labels")]
        membership = [int(line) for line in _lines(archive, "graph_indicator")]
        node_labels = [int(line) for line in _lines(archive, "node_labels")]
        attributes = [
            [float(value) for value in row.split(",")]
            for row in _lines(archive, "node_attributes")
        ]
        edge_rows = _lines(archive, "A")
    if not (len(membership) == len(node_labels) == len(attributes)):
        raise ValueError(
            "graph membership, node attributes, and node labels must have the same length"
        )

    features = [[] for _ in graph_labels]
    categories = [[] for _ in graph_labels]
    sources = [[] for _ in graph_labels]
    targets = [[] for _ in graph_labels]
    node_indices = []
    for graph_id, label, row in zip(membership, node_labels, attributes):
        if not 1 <= graph_id <= len(graph_labels):
            raise ValueError(f"invalid graph indicator {graph_id}")
        graph_index = graph_id - 1
        node_indices.append((graph_index, len(features[graph_index])))
        features[graph_index].append(row)
        categories[graph_index].append(label)

    for row in edge_rows:
        source, target = map(int, row.split(","))
        if not (1 <= source <= len(node_indices) and 1 <= target <= len(node_indices)):
            raise ValueError(f"edge outside the node table: {row}")
        graph, local_source = node_indices[source - 1]
        target_graph, local_target = node_indices[target - 1]
        if graph != target_graph:
            raise ValueError(f"edge crosses graph boundaries: {row}")
        sources[graph].append(local_source)
        targets[graph].append(local_target)

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        staging = Path(directory) / output.name
        with staging.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(_FIELDS)
            for index, label in enumerate(graph_labels):
                writer.writerow(
                    [
                        f"tu-{index + 1}",
                        len(features[index]),
                        json.dumps(features[index], allow_nan=False),
                        json.dumps(categories[index]),
                        json.dumps([sources[index], targets[index]]),
                        label,
                    ]
                )
        # Validate shape, label ranges, and graph symmetry through the training parser.
        count = len(read_graphs(staging, labeled=True))
        staging.replace(output)
    return count
