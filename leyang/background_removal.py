from rembg import remove
import cv2
import numpy as np


def remove_background(image):
    """
    Remove image background using U²-Net..

    Parameters:
        image (numpy.ndarray): Input BGR image

    Returns:
        numpy.ndarray: Background removed image (BGR)
    """

    # Check if input image is exist
    if image is None:
        raise ValueError("Input image is None.")

    # Convert BGR → RGB
    # resize的OpenCV是 Blue, Green， Red, U²-Net需要Red, Green, Blue
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Remove background (RGBA) - A是Alpha (透明度)
    output = remove(rgb_image)

    # Convert RGBA → BGR
    # 有些图片输出RGB，有些输出RGBA
    # 判断有没有4个channel，有4个channel代表是RGBA，否则就是RGB
    if output.shape[2] == 4:

        # R G B A
        # 0 1 2 3
        # 3代表Alpha
        alpha = output[:, :, 3]

        # White background
        # 建立白色背景，因为不要透明背景
        white = np.ones_like(output[:, :, :3]) * 255

        # 要RGB (香蕉)，不要Alpha
        foreground = output[:, :, :3]

        # Normalize alpha
        # 255 -> 1
        # 0 -> 0
        alpha = alpha[:, :, np.newaxis] / 255.0

        # Formula
        # 得到白底的照片
        result = foreground * alpha + white * (1 - alpha)

        # OpenCV的图片必须要 0-255 (8个bits)
        result = result.astype(np.uint8)

    else:
        result = output

    # RGB → BGR
    # 再转回来，后面的technique都需要BGR
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    return result