import cv2 as cv


def resize_image(image, width=224, height=224):
    resized_image = cv.resize(image, (width, height))

    return resized_image