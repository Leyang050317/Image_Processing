import cv2


def refine_mask(
    mask,
    kernel_size=5
):
    """
    Refine a binary mask using:

    1. Morphological Opening
       Removes small isolated foreground noise.

    2. Morphological Closing
       Fills small holes and gaps inside the banana region.
    """

    if mask is None:
        raise ValueError("Input mask cannot be None.")

    # Elliptical structuring element
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    # -----------------------------------------------------
    # Opening
    # -----------------------------------------------------

    opened = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # -----------------------------------------------------
    # Closing
    # -----------------------------------------------------

    closed = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel
    )

    return closed