import cv2
import numpy as np


def refine_main_contour(
    mask,
    opening_kernel_size=3,
    closing_kernel_size=3
):
    """
    Refine the outer banana boundary.

    A mild morphological opening is first applied to reduce
    thin attached protrusions. The largest external contour
    is then retained and filled.

    Parameters
    ----------
    mask : numpy.ndarray
        Binary banana mask.

    opening_kernel_size : int
        Kernel used for conservative removal of narrow
        connected protrusions.

    closing_kernel_size : int
        Kernel used to smooth small boundary gaps.

    Returns
    -------
    numpy.ndarray
        Refined binary banana mask.
    """

    if mask is None:
        raise ValueError("Input mask cannot be None.")

    binary_mask = np.where(
        mask > 0,
        255,
        0
    ).astype(np.uint8)

    # -----------------------------------------------------
    # Mild opening
    # Helps remove narrow attached debris
    # -----------------------------------------------------

    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            opening_kernel_size,
            opening_kernel_size
        )
    )

    opened = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        opening_kernel,
        iterations=1
    )

    # -----------------------------------------------------
    # Find external contours
    # -----------------------------------------------------

    contours, _ = cv2.findContours(
        opened,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return binary_mask.copy()

    # Keep largest contour
    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    contour_mask = np.zeros_like(
        binary_mask
    )

    cv2.drawContours(
        contour_mask,
        [largest_contour],
        contourIdx=-1,
        color=255,
        thickness=cv2.FILLED
    )

    # -----------------------------------------------------
    # Mild closing to smooth boundary
    # -----------------------------------------------------

    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            closing_kernel_size,
            closing_kernel_size
        )
    )

    refined_mask = cv2.morphologyEx(
        contour_mask,
        cv2.MORPH_CLOSE,
        closing_kernel,
        iterations=1
    )

    return refined_mask