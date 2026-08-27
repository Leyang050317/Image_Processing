import cv2 as cv
import numpy as np


def apply_white_balance(image):
    """
    Apply Gray-World white balance to reduce colour cast.
    """

    if image is None:
        raise ValueError("Input image is None.")

    image_float = image.astype(np.float32)
    b, g, r = cv.split(image_float)

    b_avg = np.mean(b)
    g_avg = np.mean(g)
    r_avg = np.mean(r)
    gray_avg = (b_avg + g_avg + r_avg) / 3

    epsilon = 1e-6
    b = b * (gray_avg / max(b_avg, epsilon))
    g = g * (gray_avg / max(g_avg, epsilon))
    r = r * (gray_avg / max(r_avg, epsilon))

    balanced = cv.merge((b, g, r))

    return np.clip(balanced, 0, 255).astype(np.uint8)
