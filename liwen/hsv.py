import cv2


def convert_to_hsv(image):
    """Convert a BGR image to HSV for colour-based surface analysis."""
    if image is None:
        raise ValueError("Input image is None.")

    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
