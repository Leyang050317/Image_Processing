from rembg import remove
import cv2
import numpy as np


def remove_background(image):
    """
    Remove image background using U2-Net and place the foreground on white.
    """

    if image is None:
        raise ValueError("Input image is None.")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = remove(rgb_image)

    if output.shape[2] == 4:
        alpha = output[:, :, 3]
        foreground = output[:, :, :3]
        white = np.ones_like(foreground) * 255

        alpha = alpha[:, :, np.newaxis] / 255.0
        result = foreground * alpha + white * (1 - alpha)
        result = result.astype(np.uint8)
    else:
        result = output

    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
