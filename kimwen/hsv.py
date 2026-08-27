import cv2 as cv


def convert_to_hsv(image):
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    return hsv_image