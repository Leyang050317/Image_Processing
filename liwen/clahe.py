import cv2


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Enhance local contrast on the LAB lightness channel."""
    if image is None:
        raise ValueError("Input image is None.")

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    enhanced_lightness = clahe.apply(lightness)
    enhanced_lab = cv2.merge((enhanced_lightness, a_channel, b_channel))

    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
