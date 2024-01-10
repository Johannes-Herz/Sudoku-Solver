from src.digit_functions import applyFunction, loadDigits

import numpy as np
import cv2 as cv

SRC = "C:\dev\Sudoku-Solver\data\digits"
DST = "C:\dev\Sudoku-Solver\data\digits\preprocessed"

def preprocess(image: np.ndarray) -> np.ndarray:    
    
    thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    image = cv.bitwise_and(thresh, thresh)
    image = ((255 - image) / 255)

    return image.astype(np.uint8)

if __name__ == '__main__':
    data = applyFunction(SRC, DST, preprocess, np.uint8)
    print(data[0][0].shape)
    print(data[0][1].shape)
    print(data[1][0].shape)
    print(data[1][1].shape)

    data = loadDigits(DST)
    print(data[0][0].shape)
    print(data[0][1].shape)
    print(data[1][0].shape)
    print(data[1][1].shape)