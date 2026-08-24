import cv2
import numpy as np


def select_safe_mask(
    largest_component_mask,
    contour_mask,
    min_retained_ratio=0.85,
    min_foreground_ratio=0.03,
    max_foreground_ratio=0.80
):
    """
    Select the safer mask between:

    1. Largest Connected Component mask
    2. Contour-refined mask

    The contour-refined mask is accepted only if it retains
    a reasonable amount of foreground and does not remove
    too much of the previous banana region.

    Parameters
    ----------
    largest_component_mask : numpy.ndarray
        Safer mask before contour refinement.

    contour_mask : numpy.ndarray
        Mask after contour refinement.

    min_retained_ratio : float
        Minimum fraction of the previous foreground area that
        must remain after contour refinement.

        Example:
        0.85 = contour refinement must retain at least 85%.

    min_foreground_ratio : float
        Minimum portion of the image that should remain foreground.

    max_foreground_ratio : float
        Maximum portion of the image that should remain foreground.

    Returns
    -------
    selected_mask : numpy.ndarray
        Mask selected for background removal.

    decision : str
        Description of which mask was selected.
    """

    if largest_component_mask is None:
        raise ValueError(
            "largest_component_mask cannot be None."
        )

    if contour_mask is None:
        raise ValueError(
            "contour_mask cannot be None."
        )

    # -----------------------------------------------------
    # Convert to binary
    # -----------------------------------------------------

    safe_mask = np.where(
        largest_component_mask > 0,
        255,
        0
    ).astype(np.uint8)

    refined_mask = np.where(
        contour_mask > 0,
        255,
        0
    ).astype(np.uint8)

    # -----------------------------------------------------
    # Calculate areas
    # -----------------------------------------------------

    safe_area = cv2.countNonZero(
        safe_mask
    )

    refined_area = cv2.countNonZero(
        refined_mask
    )

    total_pixels = (
        refined_mask.shape[0]
        * refined_mask.shape[1]
    )

    # If the safer mask is already empty,
    # there is nothing useful to compare.
    if safe_area == 0:

        return (
            refined_mask,
            "Contour mask used: previous mask was empty."
        )

    # -----------------------------------------------------
    # Area retention
    # -----------------------------------------------------

    retained_ratio = (
        refined_area / safe_area
    )

    foreground_ratio = (
        refined_area / total_pixels
    )

    # -----------------------------------------------------
    # Check 1:
    # Did contour refinement remove too much?
    # -----------------------------------------------------

    if retained_ratio < min_retained_ratio:

        return (
            safe_mask.copy(),
            (
                "Fallback to Largest Connected Component: "
                f"contour retained only "
                f"{retained_ratio:.1%} of previous mask."
            )
        )

    # -----------------------------------------------------
    # Check 2:
    # Final foreground suspiciously small
    # -----------------------------------------------------

    if foreground_ratio < min_foreground_ratio:

        return (
            safe_mask.copy(),
            (
                "Fallback to Largest Connected Component: "
                f"foreground too small "
                f"({foreground_ratio:.1%} of image)."
            )
        )

    # -----------------------------------------------------
    # Check 3:
    # Final foreground suspiciously large
    # -----------------------------------------------------

    if foreground_ratio > max_foreground_ratio:

        return (
            safe_mask.copy(),
            (
                "Fallback to Largest Connected Component: "
                f"foreground too large "
                f"({foreground_ratio:.1%} of image)."
            )
        )

    # -----------------------------------------------------
    # Refined mask appears reasonable
    # -----------------------------------------------------

    return (
        refined_mask,
        (
            "Contour mask accepted: "
            f"retained {retained_ratio:.1%} "
            f"of previous foreground."
        )
    )