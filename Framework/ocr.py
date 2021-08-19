import numpy as np
import cv2
import math
import os
import re

import settings




def get_text_from_image(img):
    """
    Function that performs ocr on an image
    works best when the background is white and the text is black (binary image)

    :param img: image being processed for ocr
    :return: Text which appears on the image
    """
    img_output_loc = settings.TESSERACT_LOCATION + "test.png"
    cv2.imwrite(img_output_loc, img)
    cmd = settings.TESSERACT_LOCATION + "tesseract.exe " + settings.TESSERACT_LOCATION + "test.png "
    cmd = cmd + settings.TESSERACT_LOCATION + "test --psm 6"
    os.system(cmd)  # currently using os call for tesseract.
    ocr_result_loc = settings.TESSERACT_LOCATION + "test.txt"
    ocr_file = open(ocr_result_loc, "r", encoding='utf-8')  # read the results
    ocr_result = ocr_file.read()
    return ocr_result




def preprocess_for_ocr(img):
    """
    Function which turn the image into a binary black/white
    image so Tesseract's ocr works better on it

    :param img: image (frame of a video) that needs to be pre-processed
    :return: pre-processed image
    """

    threshold = 50
    maximum_value = 255

    # turn the image into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # threshhold
    ret, bin = cv2.threshold(gray, threshold, maximum_value, cv2.THRESH_BINARY)
    # closing
    kernel = np.ones((3, 3), np.uint8) # 3x3 kernel filled with 1
    closing = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel)
    # invert black/white
    inv = cv2.bitwise_not(closing)
    return inv

