import cv2


def resize_image(image, width=250, height=250):
    """Resize an input BGR image to the standard LW working size."""
    if image is None:
        raise ValueError("Input image is None.")

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
