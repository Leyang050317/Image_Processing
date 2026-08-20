import cv2


def apply_bilateral_filter(image, diameter=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filtering to reduce noise while preserving edges.

    Parameters:
        image (numpy.ndarray): Input BGR image
        diameter (int): Pixel neighborhood diameter
        sigma_color (float): Filter sigma in the colour space
        sigma_space (float): Filter sigma in the coordinate space

    Returns:
        numpy.ndarray: Noise-reduced BGR image
    """

    if image is None:
        raise ValueError("Input image is None.")

    filtered_image = cv2.bilateralFilter(
        image,
        diameter,
        sigma_color,
        sigma_space
    )

    return filtered_image
