import cv2
import numpy as np


def calculate_blemish_ratio(banana_mask, blemish_mask):
    """Return blemish area, banana surface area, and their percentage ratio."""
    if banana_mask is None or blemish_mask is None:
        raise ValueError("Banana and blemish masks are required.")

    banana_area = cv2.countNonZero(banana_mask)
    blemish_inside_banana = cv2.bitwise_and(blemish_mask, banana_mask)
    blemish_area = cv2.countNonZero(blemish_inside_banana)
    blemish_ratio = (blemish_area / banana_area * 100.0) if banana_area else 0.0

    return {
        "banana_surface_area_px": int(banana_area),
        "blemish_area_px": int(blemish_area),
        "blemish_ratio_percent": float(blemish_ratio),
    }


def calculate_glcm_features(
    bgr_image,
    banana_mask,
    distances=(1,),
    angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    levels=32,
):
    """Calculate mean GLCM properties from the banana surface region.

    The image is cropped to the banana bounding box. Pixels outside the mask
    are filled with the mean valid intensity, preventing the external
    background from dominating texture co-occurrences.
    """
    if bgr_image is None or banana_mask is None:
        raise ValueError("BGR image and banana mask are required.")
    if levels < 2 or levels > 256:
        raise ValueError("levels must be between 2 and 256.")

    try:
        from skimage.feature import graycomatrix, graycoprops
    except ImportError as error:
        raise ImportError(
            "GLCM analysis requires scikit-image. Install it with: "
            "python -m pip install scikit-image"
        ) from error

    contours, _ = cv2.findContours(banana_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"glcm_contrast": 0.0, "glcm_homogeneity": 0.0, "glcm_energy": 0.0, "glcm_correlation": 0.0}

    x, y, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    gray_crop = gray_image[y:y + height, x:x + width].copy()
    mask_crop = banana_mask[y:y + height, x:x + width] > 0

    valid_pixels = gray_crop[mask_crop]
    if valid_pixels.size < 2:
        return {"glcm_contrast": 0.0, "glcm_homogeneity": 0.0, "glcm_energy": 0.0, "glcm_correlation": 0.0}

    gray_crop[~mask_crop] = int(np.mean(valid_pixels))
    quantized = (gray_crop.astype(np.uint16) * levels // 256).clip(0, levels - 1).astype(np.uint8)
    glcm = graycomatrix(
        quantized, distances=distances, angles=angles, levels=levels,
        symmetric=True, normed=True,
    )

    return {
        "glcm_contrast": float(np.mean(graycoprops(glcm, "contrast"))),
        "glcm_homogeneity": float(np.mean(graycoprops(glcm, "homogeneity"))),
        "glcm_energy": float(np.mean(graycoprops(glcm, "energy"))),
        "glcm_correlation": float(np.mean(graycoprops(glcm, "correlation"))),
    }


def calculate_surface_features(bgr_image, banana_mask, blemish_mask, **glcm_parameters):
    """Combine surface-area and GLCM texture measurements for LW analysis."""
    features = calculate_blemish_ratio(banana_mask, blemish_mask)
    features.update(calculate_glcm_features(bgr_image, banana_mask, **glcm_parameters))
    return features
