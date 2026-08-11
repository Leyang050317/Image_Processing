import cv2


def convert_to_hsv(image):
    """
    Convert BGR image to HSV color space.

    Parameters:
        image (numpy.ndarray): Input BGR image

    Returns:
        numpy.ndarray: HSV image
    """

    if image is None:
        raise ValueError("Input image is None.")

    # Convert BGR → HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    return hsv_image