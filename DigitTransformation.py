from src.digit_functions import applyFunction, loadDigits
from src.feature_functions import pixelFeatureFunction, pixelDensityFeatureFunction, imageGradientDensityFeatureFunction

import numpy as np

SRC = "C:\dev\Sudoku-Solver\data\digits\preprocessed"

def performPixelTransformation():
    DST = "C:\dev\Sudoku-Solver\data\digits\\transformed\pixel"

    print("(+) Perform pixel transformation...")
    data = applyFunction(SRC, DST, pixelFeatureFunction, dtype=np.uint8)
    print(data[0][0].shape)
    print(data[0][1].shape)
    print(data[1][0].shape)
    print(data[1][1].shape)

    data = loadDigits(DST)
    print(data[0][0].shape)
    print(data[0][1].shape)
    print(data[1][0].shape)
    print(data[1][1].shape)

    print("(+) Pixel transformation finished")

def performPixelDensityTransformation():
    DST = "C:\dev\Sudoku-Solver\data\digits\\transformed\pixelDensity"

    print("(+) Perform pixel density transformation...")
    data = applyFunction(SRC, DST, pixelDensityFeatureFunction, dtype=np.float32)
    print(data[0][0].shape)
    print(data[0][1].shape)
    print(data[1][0].shape)
    print(data[1][1].shape)

    data = loadDigits(DST)
    print(data[0][0].shape)
    print(data[0][1].shape)
    print(data[1][0].shape)
    print(data[1][1].shape)

    print("(+) Pixel Density transformation finished")

def performImageGradientDensityTransformation():
    DST = "C:\dev\Sudoku-Solver\data\digits\\transformed\imageGradientDensity"

    print("(+) Perform image gradient density transformation...")
    data = applyFunction(SRC, DST, imageGradientDensityFeatureFunction, dtype=np.float32)
    print(data[0][0].shape) 
    print(data[0][1].shape)
    print(data[1][0].shape)
    print(data[1][1].shape)

    data = loadDigits(DST)
    print(data[0][0].shape)
    print(data[0][1].shape)
    print(data[1][0].shape)
    print(data[1][1].shape)

    print("(+) Image Gradient Density transformation finished")

if __name__ == '__main__':
    performPixelTransformation()
    print('\n')
    performPixelDensityTransformation()
    print('\n')
    performImageGradientDensityTransformation()