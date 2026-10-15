
import numpy as np
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
)


BRATS_REGIONS = {
    "ET": [4],
    "TC": [1, 4],
    "WT": [1, 2, 4],
}


def dice_score(pred, target):
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)

    if pred.shape != target.shape:
        raise ValueError("Prediction and target shapes differ.")

    intersection = np.logical_and(pred, target).sum()
    denominator = pred.sum() + target.sum()

    if denominator == 0:
        return 1.0

    return float(2.0 * intersection / denominator)


def hd95_score(pred, target, spacing=(1.0, 1.0, 1.0)):
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)

    if pred.shape != target.shape:
        raise ValueError("Prediction and target shapes differ.")

    if pred.ndim != 3:
        raise ValueError("HD95 expects 3D masks.")

    if len(spacing) != 3 or any(s <= 0 for s in spacing):
        raise ValueError("Spacing must contain 3 positive values.")

    if not pred.any() and not target.any():
        return 0.0

    if not pred.any() or not target.any():
        return float("nan")

    structure = generate_binary_structure(3, 1)

    pred_surface = pred ^ binary_erosion(
        pred, structure=structure, border_value=0
    )

    target_surface = target ^ binary_erosion(
        target, structure=structure, border_value=0
    )

    pred_distance = distance_transform_edt(
        ~pred_surface, sampling=spacing
    )

    target_distance = distance_transform_edt(
        ~target_surface, sampling=spacing
    )

    distances = np.concatenate([
        target_distance[pred_surface],
        pred_distance[target_surface],
    ])

    return float(np.percentile(distances, 95))


def evaluate_brats(pred, target, spacing=(1.0, 1.0, 1.0)):
    """
    Calculate ET, TC, and WT Dice and HD95.

    Inputs must use BraTS labels 0, 1, 2, 4.
    """

    pred = np.asarray(pred)
    target = np.asarray(target)

    if pred.shape != target.shape:
        raise ValueError("Prediction and target shapes differ.")

    results = {}

    for region, labels in BRATS_REGIONS.items():
        pred_mask = np.isin(pred, labels)
        target_mask = np.isin(target, labels)

        results[f"{region}_Dice"] = dice_score(
            pred_mask, target_mask
        )

        results[f"{region}_HD95"] = hd95_score(
            pred_mask, target_mask, spacing
        )

    return results
