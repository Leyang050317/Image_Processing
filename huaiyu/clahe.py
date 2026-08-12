# huaiyu/clahe.py

import cv2 as cv


def apply_clahe(image):
    # Convert BGR to LAB
    lab_image = cv.cvtColor(
        image,
        cv.COLOR_BGR2LAB
    )

    l, a, b = cv.split(lab_image)

    clahe = cv.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l)

    enhanced_lab = cv.merge(
        (enhanced_l, a, b)
    )

    enhanced_image = cv.cvtColor(
        enhanced_lab,
        cv.COLOR_LAB2BGR
    )

    return enhanced_image