import cv2


def apply_bilateral_filter(
    image,
    diameter=5,
    sigma_color=50,
    sigma_space=50

):
    """
    Apply edge-preserving bilateral filtering.

    Bilateral filtering reduces image noise while preserving
    boundaries that may later be useful for banana segmentation.

    Parameters
    ----------
    image : numpy.ndarray
        Input BGR image.

    diameter : int
        Diameter of the pixel neighbourhood.

    sigma_color : float
        Controls filtering of pixels with different intensities.

    sigma_space : float
        Controls filtering based on spatial distance.

    Returns
    -------
    numpy.ndarray
        Bilaterally filtered image.
    """

    if image is None:
        raise ValueError("Input image cannot be None.")

    return cv2.bilateralFilter(
        image,
        d=diameter,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    )