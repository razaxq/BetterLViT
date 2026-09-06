"""Choose checkpoints by a preregistered per-image validation metric."""
import math


def is_improvement(history, metric):
    if metric not in ("dice", "iou"):
        raise ValueError("Selection metric must be dice or iou")
    current = history[-1]
    key = "val_" + metric
    score = current[key]
    if not math.isfinite(score):
        raise ValueError("Non-finite validation selection score")
    # Preserve the historical first-five-epoch exclusion for all paired arms.
    if current["epoch"] <= 5:
        return False
    previous = [row[key] for row in history[:-1] if row["epoch"] > 5]
    return score > max(previous, default=0.0)
