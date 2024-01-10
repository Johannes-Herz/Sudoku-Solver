
from src.feature_functions import *
from src.sudoku_solver_functions import *
from Models import PixelModel, PixelDensityModel, ImageGradientDensityModel

import cv2 as cv
import torch
from collections.abc import Callable

_type_ = "pixel"
# _type_ = "pixelGradient"
# _type_ = "imageGradientDensity"

MODEL = None
FEATURE_FUNCTION = None

match(_type_): 
    case 'pixel':
        MODEL = torch.load("./data/models/pixel.pth")
        FEATURE_FUNCTION = pixelFeatureFunction
    case 'pixelDensity':
        MODEL = torch.load("./data/models/pixelDensity.pth")
        FEATURE_FUNCTION = pixelDensityFeatureFunction
    case 'imageGradientDensity':
        MODEL = torch.load("./data/models/imageGradientDensity.pth")
        FEATURE_FUNCTION = imageGradientDensityFeatureFunction


def testImage(pathToImage: str, model: torch.nn.Module, featureFunction: Callable): 

    frame = cv.imread(pathToImage)

    if frame is None: 
        cv.destroyAllWindows()
        return

    output = solveSudokuInImage(frame, model, featureFunction)

    if output is None: 
        cv.imshow('frame', frame)
    else: 
        cv.imshow('frame', frame)
        cv.imshow('output', output)

    cv.waitKey(10000)
    cv.destroyAllWindows()

def testVideo(pathToVideo: str, model: torch.nn.Module, featureFunction: Callable): 

    vid = cv.VideoCapture(0)

    if pathToVideo is not None:
        vid.open(pathToVideo)
    
    while True: 

        ret, frame = vid.read()

        if frame is None: 
            cv.destroyAllWindows()
            return
        
        frame = cv.resize(frame, (int(frame.shape[1]/3), int(frame.shape[0]/3)))
        output = solveSudokuInImage(frame, model, featureFunction)

        if output is None: 
            cv.imshow('frame', frame)
        else: 
            cv.imshow('frame', frame)
            cv.imshow('output', output)

        if cv.waitKey(100) & 0xFF == ord('q'):
            break
    
    cv.destroyAllWindows()


if __name__ == '__main__':
    testImage('C:\dev\Sudoku-Solver\demo.jpg', MODEL, FEATURE_FUNCTION)
    # testImage('C:\dev\Sudoku-Solver\data\sudokus\plain\image202.jpg', MODEL, FEATURE_FUNCTION)
    # testVideo('C:\dev\Sudoku-Solver\demo.mp4', MODEL, FEATURE_FUNCTION)
    pass