# huaiyu/background_removal.py

import cv2 as cv
import numpy as np
from rembg import remove


def remove_background(image):
    # Convert OpenCV BGR image to RGB
    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Perform background removal
    result = remove(rgb_image)

    # rembg may return an RGBA image
    if result.shape[2] == 4:
        rgb = result[:, :, :3]
        alpha = result[:, :, 3]

        # Convert alpha channel into binary mask
        _, mask = cv.threshold(
            alpha,
            127,
            255,
            cv.THRESH_BINARY
        )

    else:
        rgb = result
        mask = np.ones(
            result.shape[:2],
            dtype=np.uint8
        ) * 255

    # Convert RGB back to BGR
    foreground = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)

    return foreground, mask