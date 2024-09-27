# segment.py
import cv2
import numpy as np
from tensorflow import keras
import os
import sys

def refineImg(img):
    _, thresh = cv2.threshold(cv2.convertScaleAbs(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def largestCountour(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    if num_contours != 1:
        return None
    return max(contours, key=cv2.contourArea) if contours else None

def fit(contour):
    return cv2.fitEllipse(contour) if contour is not None else None

def segment(test_img):
    model = keras.models.load_model('best_model.h5', compile=False)
    original_height, original_width, _ = test_img.shape

    test_img = cv2.resize(test_img, (256, 256))
    test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
    test_img = np.expand_dims(test_img, axis=0)

    prediction = model.predict(test_img)
    prediction_img = prediction.reshape(256, 256)
    prediction_img = np.expand_dims(prediction_img, axis=-1)

    # resize image back to original shape before displaying
    test_img = cv2.resize(test_img[0], (original_width, original_height))
    prediction_img = cv2.resize(prediction_img, (original_width, original_height))

    binaryImg = refineImg(prediction_img)
    ellipse = fit(largestCountour(binaryImg))

    if ellipse:

        cv2.ellipse(test_img, ellipse, (0, 0, 255), 2)

        semi_axes_b, semi_axes_a = ellipse[1]
        if semi_axes_b > semi_axes_a:
            semi_axes_b += semi_axes_a
            semi_axes_a = semi_axes_b - semi_axes_a
            semi_axes_b -= semi_axes_a

        angle = ellipse[2]
        if angle < 90:
            angle += 90
        else:
            angle -= 90
    else:
        print("No valid ellipse found.")
        return None

    return test_img, semi_axes_a, semi_axes_b, angle

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
