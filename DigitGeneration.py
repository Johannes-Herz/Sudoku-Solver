from src.digit_functions import generateDigits, loadDigits

import os
import numpy as np

DST = "C:\dev\Sudoku-Solver\data\digits"
IMAGES_PER_DIGIT = 175000
FONT_PATHS = [f"C:\dev\Sudoku-Solver\data\\fonts\{filename}" for filename in os.listdir("C:\dev\Sudoku-Solver\data\\fonts") if filename.endswith('.ttf') or filename.endswith('.otf')]
TRAIN_TEST_SPLIT = 0.95

if __name__ == '__main__':
    data = generateDigits(DST, IMAGES_PER_DIGIT, FONT_PATHS, TRAIN_TEST_SPLIT)
    print(data[0][0].shape)
    print(data[0][1].shape)
    print(data[1][0].shape)
    print(data[1][1].shape)

    data = loadDigits(DST)
    print(data[0][0].shape)
    print(data[0][1].shape)
    print(data[1][0].shape)
    print(data[1][1].shape)