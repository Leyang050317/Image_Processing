import cv2
import numpy as np


def refine_segmentation_grabcut(
    image,
    coarse_mask,
    iterations=5
):
    """
    Refine the HSV coarse segmentation using GrabCut.

    HSV provides an initial estimate of the banana location.
    GrabCut improves foreground/background separation and
    helps recover darker banana regions.
    """

    if image is None:
        raise ValueError("Input image cannot be None.")

    if coarse_mask is None:
        raise ValueError("Coarse mask cannot be None.")

    # -----------------------------------------------------
    # Create seed regions
    # -----------------------------------------------------

    kernel = np.ones(
        (5, 5),
        dtype=np.uint8
    )

    # High-confidence foreground
    sure_foreground = cv2.erode(
        coarse_mask,
        kernel,
        iterations=1
    )

    # Expanded probable foreground area
    probable_region = cv2.dilate(
        coarse_mask,
        kernel,
        iterations=3
    )

    # -----------------------------------------------------
    # Initialise GrabCut mask
    # -----------------------------------------------------

    grabcut_mask = np.full(
        coarse_mask.shape,
        cv2.GC_BGD,
        dtype=np.uint8
    )

    # Probable foreground
    grabcut_mask[
        probable_region > 0
    ] = cv2.GC_PR_FGD

    # Original HSV regions remain probable foreground
    grabcut_mask[
        coarse_mask > 0
    ] = cv2.GC_PR_FGD

    # Strong internal region becomes definite foreground
    grabcut_mask[
        sure_foreground > 0
    ] = cv2.GC_FGD

    # -----------------------------------------------------
    # GrabCut models
    # -----------------------------------------------------

    background_model = np.zeros(
        (1, 65),
        dtype=np.float64
    )

    foreground_model = np.zeros(
        (1, 65),
        dtype=np.float64
    )

    # -----------------------------------------------------
    # Run GrabCut
    # -----------------------------------------------------

    cv2.grabCut(
        image,
        grabcut_mask,
        None,
        background_model,
        foreground_model,
        iterations,
        cv2.GC_INIT_WITH_MASK
    )

    # -----------------------------------------------------
    # Convert GrabCut classes into binary mask
    # -----------------------------------------------------

    refined_mask = np.where(
        (grabcut_mask == cv2.GC_FGD)
        |
        (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    return refined_mask