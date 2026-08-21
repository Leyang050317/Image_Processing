import cv2


def apply_median_filter(image, kernel_size=5):
    """Reduce impulse noise while retaining object edges.

    OpenCV requires a positive odd kernel size greater than one.
    """
    if image is None:
        raise ValueError("Input image is None.")
    if kernel_size <= 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer greater than 1.")

    return cv2.medianBlur(image, kernel_size)
