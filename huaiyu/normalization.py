# huaiyu/normalization.py


def normalize_image(image):
    normalized = image.astype("float32") / 255.0

    return normalized