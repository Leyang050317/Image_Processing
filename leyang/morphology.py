import cv2
import numpy as np


def refine_mask(mask, kernel_size=5):
    """
    Refine a binary mask using opening and closing operations.

    Parameters:
        mask (numpy.ndarray): Input binary mask
        kernel_size (int): Size of the morphology kernel

    Returns:
        numpy.ndarray: Refined binary mask
    """

    if mask is None:
        raise ValueError("Input mask is None.")

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    _, binary_mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    opened_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    refined_mask = cv2.morphologyEx(
        opened_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return refined_mask


def apply_mask(image, mask):
    """
    Apply a binary mask to an image and replace the background with white.

    Parameters:
        image (numpy.ndarray): Input BGR image
        mask (numpy.ndarray): Binary mask

    Returns:
        numpy.ndarray: Masked BGR image with white background
    """

    if image is None:
        raise ValueError("Input image is None.")

    if mask is None:
        raise ValueError("Input mask is None.")

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    _, binary_mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    foreground = cv2.bitwise_and(
        image,
        image,
        mask=binary_mask
    )

    white_background = np.ones_like(image) * 255
    inverse_mask = cv2.bitwise_not(binary_mask)

    background = cv2.bitwise_and(
        white_background,
        white_background,
        mask=inverse_mask
    )

    result = cv2.add(
        foreground,
        background
    )

    return result


def refine_blemish_mask(mask, kernel_size=3):
    """
    Refine blemish mask while keeping small surface spots.

    Blemishes can be tiny, so this uses closing only instead of opening.
    """

    if mask is None:
        raise ValueError("Input mask is None.")

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    _, binary_mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    refined_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return refined_mask
