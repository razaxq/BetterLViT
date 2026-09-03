"""Deterministic report semantics and coarse lung-region bases for RACE-Fuse.

The QaTa-COV19 reports use a small, repetitive vocabulary.  RACE uses the
parser only to create weak supervision for a learned text-slot head; the
parser is not part of inference.  Unmentioned image regions are deliberately
left as unknown by the RACE consistency loss.
"""

import re

import numpy as np
import torch


ZONE_NAMES = (
    "left_upper",
    "left_middle",
    "left_lower",
    "right_upper",
    "right_middle",
    "right_lower",
)


def _contains(text, pattern):
    return re.search(pattern, text) is not None


def parse_report_slots(text):
    """Return six mentioned zones plus a one-hot lesion-count class.

    The count classes are one, two, and three-or-more.  Location extraction is
    conservative: ambiguous descriptions may activate several zones, while a
    missing location activates none rather than inventing a negative label.
    """

    normalized = " ".join(str(text).lower().replace("-", " ").split())
    sides = []
    if _contains(normalized, r"\b(left|lt)\b"):
        sides.append(0)
    if _contains(normalized, r"\b(right|rt)\b"):
        sides.append(1)
    if _contains(normalized, r"\b(bilateral|both)\b") or not sides:
        sides = [0, 1]

    levels = []
    if _contains(normalized, r"\b(upper|superior|apical)\b"):
        levels.append(0)
    if _contains(normalized, r"\b(mid|middle|perihilar)\b"):
        levels.append(1)
    if _contains(normalized, r"\b(lower|inferior|basal|base)\b"):
        levels.append(2)
    if not levels:
        levels = [0, 1, 2]

    zones = torch.zeros(6, dtype=torch.float32)
    for side in sides:
        for level in levels:
            zones[side * 3 + level] = 1.0

    count = 1
    if _contains(normalized, r"\b(three|3|multiple|several)\b"):
        count = 3
    elif _contains(normalized, r"\b(two|2|double)\b"):
        count = 2
    elif _contains(normalized, r"\b(one|1|single)\b"):
        count = 1
    count_one_hot = torch.zeros(3, dtype=torch.float32)
    count_one_hot[min(max(count, 1), 3) - 1] = 1.0
    return torch.cat([zones, count_one_hot], dim=0)


def make_zone_basis(height, width):
    """Create a disjoint left/right by upper/middle/lower six-zone basis."""

    basis = np.zeros((6, int(height), int(width)), dtype=np.float32)
    x_edges = (0, int(width) // 2, int(width))
    y_edges = (0, int(height) // 3, 2 * int(height) // 3, int(height))
    for side in range(2):
        for level in range(3):
            basis[side * 3 + level,
                  y_edges[level]:y_edges[level + 1],
                  x_edges[side]:x_edges[side + 1]] = 1.0
    return basis
