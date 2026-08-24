import cv2
import numpy as np


def keep_largest_component(mask):
    """
    Keep only the largest connected foreground component.

    This removes smaller unrelated segmented objects and
    retains the primary banana region.
    """

    if mask is None:
        raise ValueError("Input mask cannot be None.")

    # Ensure binary mask
    binary_mask = np.where(
        mask > 0,
        255,
        0
    ).astype(np.uint8)

    # -----------------------------------------------------
    # Connected component analysis
    # -----------------------------------------------------

    number_of_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8
        )
    )

    # Only background exists
    if number_of_labels <= 1:
        return binary_mask.copy()

    # Ignore label 0 because label 0 is background
    component_areas = stats[
        1:,
        cv2.CC_STAT_AREA
    ]

    largest_label = (
        1 + np.argmax(component_areas)
    )

    # -----------------------------------------------------
    # Generate final mask
    # -----------------------------------------------------

    largest_component_mask = np.zeros_like(
        binary_mask
    )

    largest_component_mask[
        labels == largest_label
    ] = 255

    return largest_component_mask