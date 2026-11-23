"""Deterministic graph descriptors and pairwise distances for ENZYMES."""

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components, shortest_path
from torch.nn import functional as F

from .data import Graph

RETRIEVAL_NAMES = (
    "geometry-trim-0.4",
    "raw-trim-0.4",
    "raw-trim-0.8",
    "raw-kernel-1.5",
    "categories-kernel-1.5",
)
_FEATURE_SETS = ("raw", "geometry", "categories")
_KERNEL_WIDTHS = (0.025, 0.07, 0.15, 0.35, 0.75, 1.5)
_TRIM_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)


def _distribution_descriptor(graph):
    a, x, categories = (
        graph.adjacency.numpy(),
        graph.features.numpy(),
        graph.node_types.numpy(),
    )
    x = np.sign(x) * np.log1p(np.abs(x))
    n = len(x)
    z = np.eye(3)[categories]
    degree = a.sum(1)
    d = np.maximum(degree, 1)
    dist = shortest_path(a, directed=False, unweighted=True)
    finite = dist[np.isfinite(dist) & (dist > 0)]
    eig = np.linalg.eigvalsh(np.eye(n) - a / np.sqrt(d[:, None] * d[None, :]))
    triangles = ((a @ a) * a).sum(1) / 2
    clustering = 2 * triangles / np.maximum(degree * (degree - 1), 1)
    pair = z.T @ a @ z
    d_hist = np.bincount(degree.astype(int).clip(0, 12), minlength=13)
    paths = np.array([(dist == i).sum() for i in range(1, 9)])
    topology = np.r_[
        np.log1p(n),
        np.log1p(a.sum() / 2),
        a.mean(),
        (degree == 0).mean(),
        np.mean(degree),
        np.std(degree),
        np.max(degree),
        clustering.mean(),
        np.log1p(triangles.sum()),
        connected_components(a, return_labels=False),
        np.quantile(finite, [0, 0.25, 0.5, 0.75, 1]) if finite.size else np.zeros(5),
        np.quantile(eig, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]),
        np.log1p(z.sum(0)),
        z.mean(0),
        np.log1p(pair.ravel()),
        pair.ravel() / max(a.sum(), 1),
        d_hist / n,
        paths / n**2,
    ]

    def summarize(v):
        return np.r_[
            v.mean(0),
            v.std(0),
            np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9], axis=0).ravel(),
        ]

    return np.r_[topology, summarize(x)]


def composition(graph):
    x = graph.features.numpy()
    a = graph.adjacency.numpy()
    category = graph.node_types.numpy()
    n = len(x)
    labels = np.eye(3)[category]
    counts = x[:, 6:]
    totals = counts.sum(axis=1, keepdims=True).clip(1)
    ratio = counts / totals
    totals_by_type = labels.T @ counts
    ratio_by_type = totals_by_type / totals_by_type.sum(axis=1, keepdims=True).clip(1)
    overall = counts.sum(axis=0) / max(1, counts.sum())
    chemical = x[:, 1:6] / totals
    whole = x.sum(axis=0) / totals.sum()
    graph_size = np.r_[
        np.log1p(totals.sum()),
        np.log1p(n),
        np.log1p(a.sum() / 2),
        np.log1p((labels.T @ totals).ravel()),
        labels.mean(axis=0),
        a.mean(),
    ]
    chem_by_type = labels.T @ x[:, 1:6] / (labels.T @ totals).clip(1)
    quantile = np.quantile(
        np.column_stack((ratio, chemical)), [0.2, 0.5, 0.8], axis=0
    ).ravel()
    return np.r_[
        graph_size,
        overall,
        whole,
        ratio.mean(0),
        ratio.std(0),
        ratio_by_type.ravel(),
        chem_by_type.ravel(),
        quantile,
    ]


def _alignment_features(graph):
    x = graph.features
    a = graph.adjacency
    degree = a.sum(1, keepdim=True)
    types = F.one_hot(graph.node_types, 3).float()
    signed = x.sign() * x.abs().log1p()
    total = x[:, 6:].sum(1, keepdim=True).clamp_min(1)
    relative = torch.cat((x[:, 1:6] / total, x[:, 6:] / total), dim=1)
    inv = 1 / (degree + 1)
    hop = (
        a @ torch.cat((signed, types), dim=1) + torch.cat((signed, types), dim=1)
    ) * inv
    two = (a @ hop + hop) * inv
    dist = shortest_path(a.numpy(), directed=False, unweighted=True)
    distances = torch.tensor(
        np.column_stack([(dist == d).mean(1) for d in range(1, 7)]), dtype=torch.float32
    )
    return torch.cat(
        (
            signed * 0.65,
            relative * 2,
            types * 1.5,
            hop * 0.27,
            two * 0.24,
            distances * 1.1,
            degree.log1p() / 2,
        ),
        dim=1,
    )


def _alignment_distances(graphs, device):
    sorted_ids = sorted(range(len(graphs)), key=lambda i: len(graphs[i].features))
    blocks = []
    for start in range(0, len(graphs), 16):
        ids = sorted_ids[start : start + 16]
        samples = [_alignment_features(graphs[i]) for i in ids]
        attr = torch.nn.utils.rnn.pad_sequence(samples, batch_first=True).to(device)
        size = torch.tensor([len(s) for s in samples], device=device)
        mask = torch.arange(attr.size(1), device=device)[None, :] < size[:, None]
        blocks.append((ids, attr, mask, size))
    matrices = np.zeros((3, len(graphs), len(graphs)), dtype=np.float32)
    for ii, (i_idx, x, xm, xn) in enumerate(blocks):
        for jj, (j_idx, y, ym, yn) in enumerate(blocks[: ii + 1]):
            # Vectorize every pair in this size-sorted block.
            cost = torch.cdist(x[:, None], y[None, :]).square()
            valid = xm[:, None, :, None] & ym[None, :, None, :]
            minima_a = cost.masked_fill(~ym[None, :, None, :], float("inf")).amin(-1)
            minima_b = cost.masked_fill(~xm[:, None, :, None], float("inf")).amin(-2)
            chamfer = (
                minima_a.masked_fill(~xm[:, None, :], 0).sum(-1) / xn[:, None]
                + minima_b.masked_fill(~ym[None, :, :], 0).sum(-1) / yn[None, :]
            ) * 0.5
            kernel = (-cost / 0.75).clamp(-60, 0).exp() * valid
            marg_A = xm[:, None, :] / xn[:, None, None]
            marg_B = ym[None, :, :] / yn[None, :, None]
            u = torch.ones_like(marg_A)
            v = torch.ones_like(marg_B)
            for _ in range(22):
                u = marg_A / (kernel @ v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-20)
                v = marg_B / (kernel.transpose(-2, -1) @ u.unsqueeze(-1)).squeeze(
                    -1
                ).clamp_min(1e-20)
            ot = (u.unsqueeze(-1) * kernel * v.unsqueeze(-2) * cost).sum(dim=(-2, -1))
            mutual = chamfer + torch.abs(xn[:, None].log() - yn[None, :].log())
            vals = torch.stack((chamfer, ot, mutual)).cpu().numpy()
            for k in range(len(vals)):
                matrices[k][np.ix_(i_idx, j_idx)] = vals[k]
                if ii != jj:
                    matrices[k][np.ix_(j_idx, i_idx)] = vals[k].T
        if ii % 8 == 0:
            print("alignment block", ii + 1, "of", len(blocks), flush=True)
    for mat in matrices:
        np.fill_diagonal(mat, 0)
    return matrices


def _retrieval_features(graph):
    x = graph.features
    labels = F.one_hot(graph.node_types, 3)
    signed = x.sign() * x.abs().log1p()
    degree = graph.adjacency.sum(1, keepdim=True)
    total = x[:, 6:].sum(1, keepdim=True).clamp_min(1)
    fractions = x[:, 6:] / total
    return (
        torch.cat((signed, labels * 2), 1),
        torch.cat((signed[:, :6], fractions * 3, labels * 2, degree.log1p() * 0.4), 1),
        torch.cat((signed[:, 6:], labels * 2), 1),
    )


def _retrieval_distances(graphs, device):
    order = sorted(range(len(graphs)), key=lambda i: len(graphs[i].features))
    sets = []
    names = []
    for name in _FEATURE_SETS:
        names.extend(f"{name}-kernel-{t}" for t in _KERNEL_WIDTHS)
        names.extend(f"{name}-trim-{cut}" for cut in _TRIM_FRACTIONS)
    output = np.zeros((len(names), len(graphs), len(graphs)), dtype=np.float32)
    for start in range(0, len(order), 24):
        ids = order[start : start + 24]
        cols = tuple(
            torch.nn.utils.rnn.pad_sequence(samples, batch_first=True).to(device)
            for samples in zip(*(_retrieval_features(graphs[i]) for i in ids))
        )
        sizes = torch.tensor([len(graphs[i].features) for i in ids], device=device)
        mask = torch.arange(cols[0].shape[1], device=device)[None, :] < sizes[:, None]
        sets.append((ids, cols, sizes, mask))
    for i, (ix, xs, nx, mx) in enumerate(sets):
        for j, (jx, ys, ny, my) in enumerate(sets[: i + 1]):
            vals = []
            for x, y in zip(xs, ys):
                cost = torch.cdist(x[:, None], y[None, :]).square()
                aa = cost.masked_fill(~my[None, :, None, :], torch.inf).amin(-1)
                bb = cost.masked_fill(~mx[:, None, :, None], torch.inf).amin(-2)
                for t in _KERNEL_WIDTHS:
                    a = (-aa / t).exp().masked_fill(~mx[:, None, :], 0).sum(-1) / nx[
                        :, None
                    ]
                    b = (-bb / t).exp().masked_fill(~my[None, :, :], 0).sum(-1) / ny[
                        None, :
                    ]
                    vals.append(1 - a * b)
                aa = aa.masked_fill(~mx[:, None, :], torch.inf).sort(dim=-1).values
                bb = bb.masked_fill(~my[None, :, :], torch.inf).sort(dim=-1).values
                positions_a = torch.arange(aa.shape[-1], device=device)
                positions_b = torch.arange(bb.shape[-1], device=device)
                for cut in _TRIM_FRACTIONS:
                    na = (nx * cut).ceil()
                    nb = (ny * cut).ceil()
                    av = (
                        aa.masked_fill(
                            positions_a[None, None, :] >= na[:, None, None], 0
                        ).sum(-1)
                        / na[:, None]
                    )
                    bv = (
                        bb.masked_fill(
                            positions_b[None, None, :] >= nb[None, :, None], 0
                        ).sum(-1)
                        / nb[None, :]
                    )
                    vals.append((av + bv) * 0.5)
            values = torch.stack(vals).cpu().numpy()
            for k, value in enumerate(values):
                output[k][np.ix_(ix, jx)] = value
                if i != j:
                    output[k][np.ix_(jx, ix)] = value.T
        if i % 5 == 0:
            print(f"retrieval block {i + 1}/{len(sets)}", flush=True)
    return names, output


def _topology_distances(graphs, device):
    order = sorted(range(len(graphs)), key=lambda i: len(graphs[i].features))
    blocks = []
    for offset in range(0, len(order), 16):
        ids = order[offset : offset + 16]
        lengths = torch.tensor([len(graphs[i].features) for i in ids], device=device)
        x = torch.nn.utils.rnn.pad_sequence(
            [_retrieval_features(graphs[i])[0] for i in ids], batch_first=True
        ).to(device)
        width = x.shape[1]
        a = torch.zeros(len(ids), width, width)
        for i, index in enumerate(ids):
            matrix = graphs[index].adjacency
            a[i, : len(matrix), : len(matrix)] = matrix
        a = a.to(device)
        mask = torch.arange(width, device=device)[None, :] < lengths[:, None]
        a = a + torch.diag_embed(mask.to(a.dtype))
        a = a / a.sum(-1, keepdim=True).clamp_min(1)
        blocks.append((ids, x, a, lengths, mask))
    dimensions = 5
    out = np.zeros((dimensions, len(graphs), len(graphs)), dtype=np.float32)
    for i, (ix, x, a, nx, mx) in enumerate(blocks):
        for j, (jx, y, b, ny, my) in enumerate(blocks[: i + 1]):
            cost = torch.cdist(x[:, None], y[None, :]).square()
            vals = []
            for tau in (0.35, 1.20):
                m = mx[:, None, :, None] & my[None, :, None, :]
                initial = (-cost / tau).exp() * m
                # Reward neighbors with their own plausible corresponding nodes.
                post = a[:, None] @ initial @ b[None, :].transpose(-2, -1)
                reinforced = (initial * (0.12 + post)).sqrt()
                strong = (
                    reinforced
                    * (
                        0.1 + a[:, None] @ reinforced @ b[None, :].transpose(-2, -1)
                    ).sqrt()
                )
                for matrix in (reinforced, strong):
                    cover_x = matrix.amax(-1).sum(-1) / nx[:, None]
                    cover_y = matrix.amax(-2).sum(-1) / ny[None, :]
                    vals.append(1 - (cover_x * cover_y).sqrt())
            cross = 1 - vals[-1]
            vals.append(
                1
                - cross
                * torch.exp(-0.3 * (nx[:, None].log() - ny[None, :].log()).abs())
            )
            values = torch.stack(vals).cpu().numpy()
            for k, value in enumerate(values):
                out[k][np.ix_(ix, jx)] = value
                if i != j:
                    out[k][np.ix_(jx, ix)] = value.T
        if i % 8 == 0:
            print("topology block", i + 1, "/", len(blocks), flush=True)
    return out


def graph_descriptor(graph: Graph) -> np.ndarray:
    """Summarize graph structure, attribute distributions, and node composition."""
    return np.r_[_distribution_descriptor(graph), composition(graph)]


@torch.inference_mode()
def compute_distances(graphs: list[Graph], device: torch.device) -> np.ndarray:
    """Return the distance channels used by the geometry classifier."""
    names, retrieval = _retrieval_distances(graphs, device)
    selected = [names.index(name) for name in RETRIEVAL_NAMES]
    return np.concatenate(
        (
            retrieval[selected],
            _alignment_distances(graphs, device),
            _topology_distances(graphs, device)[[0, 2]],
        )
    )
