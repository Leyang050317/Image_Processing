import numpy as np


def normalize_image(image):
    normalized_image = image.astype(np.float32) / 255.0

    return normalized_image