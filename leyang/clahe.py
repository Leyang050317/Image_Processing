import cv2


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE to the Lightness channel in LAB color space..

    Parameters:
        image (numpy.ndarray): Input BGR image
        clip_limit (float): Contrast limiting threshold
        tile_grid_size (tuple): Size of grid for histogram equalization

    Returns:
        numpy.ndarray: Contrast-enhanced BGR image
    """

    if image is None:
        raise ValueError("Input image is None.")

    # Convert BGR to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split LAB channels
    l, a, b = cv2.split(lab)

    # Create CLAHE object
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,        # 增强程度
        tileGridSize=tile_grid_size  # 把图片分成小块 (8 x 8)
    )

    # Apply CLAHE to Lightness channel
    l = clahe.apply(l)

    # Merge channels
    lab = cv2.merge((l, a, b))

    # Convert LAB back to BGR
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return result