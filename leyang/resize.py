import cv2
import os

def resize_image(image, width=250, height=250):
    """
    Resize an image..

    Parameters:
        image (numpy.ndarray): Input image
        width (int): Target width
        height (int): Target height

    Returns:
        numpy.ndarray: Resized image
    """

    if image is None:
        raise ValueError("Input image is None.")

    resized_image = cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    return resized_image