import numpy as np
import torch
from torch import Tensor


def compute_objectives_pf_distances(
    pf_points: Tensor, y0_min: float, y0_max: float, y1_min: float, y1_max: float, n: int
) -> Tensor:
    y0_len = y0_max - y0_min
    y0_start = y0_min + y0_len / (n * 2)
    y0_end = y0_max - y0_len / (n * 2)

    y1_len = y1_max - y1_min
    y1_start = y1_min + y1_len / (n * 2)
    y1_end = y1_max - y1_len / (n * 2)

    y0s = np.linspace(y0_start, y0_end, n, dtype=np.float32)
    y1s = np.linspace(y1_start, y1_end, n, dtype=np.float32)

    distances = torch.zeros(n, n)
    for i, y0 in enumerate(y0s):
        for j, y1 in enumerate(y1s):
            y = torch.tensor([y0, y1])
            distances[i][j] = compute_pf_distance(pf_points, y)

    max_distance = torch.max(distances[distances.isfinite()])
    distances = distances / max_distance
    distances[distances.isnan()] = -1.0
    return distances


def compute_pf_distance(pf_points: Tensor, y: Tensor) -> Tensor:
    """Compute the distance from a point y to a piecewise-linear Pareto front.

    The Pareto front is approximated as a polyline: the ordered sequence of
    ``pf_points`` defines consecutive line segments, and the distance returned
    is the minimum Euclidean distance from ``y`` to any of those segments.

    For each segment [A, B] the closest point on the segment to ``y`` is found
    via orthogonal projection:

        t = dot(y - A, B - A) / ||B - A||²

    ``t`` is clamped to [0, 1] so that the closest point is constrained to the
    segment rather than the infinite line through A and B. This ensures correct
    distances when ``y`` lies "outside" the extent of the front (i.e. beyond
    either endpoint).

    Args:
        pf_points: Pareto front points of shape ``(k, n)``, ordered along the
            front. Adjacent points define the segments of the polyline.
        y: Query point of shape ``(n,)`` whose distance to the front is sought.

    Returns:
        Scalar tensor containing the minimum distance from ``y`` to the front.
    """
    if len(pf_points) == 1:
        return (y - pf_points[0]).norm()

    pf_first = pf_points[:-1, :]
    pf_second = pf_points[1:, :]
    d = pf_second - pf_first
    t = ((y - pf_first) * d).sum(dim=1) / (d * d).sum(dim=1)
    closest = pf_first + t.clamp(0, 1).unsqueeze(1) * d

    # Clamp at 0 so that points below the PF have a distance of 0 to it.
    distances = torch.clamp(y - closest, min=0).norm(dim=1)
    return torch.min(distances)


def compute_pf_distance_nd(pf_points: Tensor, y: Tensor) -> Tensor:
    """Compute the distance from a point y to a set of points in the Pareto front.

    Args:
        pf_points: Pareto front points of shape ``(k, n)``.
        y: Query point of shape ``(n,)`` whose distance to the front is sought.

    Returns:
        Scalar tensor containing the minimum distance from ``y`` to the front.
    """

    distances = torch.cdist(pf_points, y.unsqueeze(0))
    return distances.min()
