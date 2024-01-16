
from src.feature_functions import *
from src.sudoku_solver_functions import *
from Models import PixelModel, PixelDensityModel, ImageGradientDensityModel
from src.sudoku_functions import loadSudokus

import cv2 as cv
import torch
from collections.abc import Callable

modeltypes = ['pixel']
#modeltypes = ['pixel', 'pixelGradient']
#modeltypes = ['pixel', 'pixelGradient', 'imageGradientDensity']

MODELS = []
FEATURE_FUNCTIONS = []

for type in modeltypes:

    match(type): 
        case 'pixel':
            MODELS.append(torch.load("./models/pixel.pth"))
            FEATURE_FUNCTIONS.append(pixelFeatureFunction)
        case 'pixelDensity':
            MODELS.append(torch.load("./models/pixelDensity.pth"))
            FEATURE_FUNCTIONS.append(pixelDensityFeatureFunction)
        case 'imageGradientDensity':
            MODELS.append(torch.load("./models/imageGradientDensity.pth"))
            FEATURE_FUNCTIONS.append(imageGradientDensityFeatureFunction)


def testImage(Image: np.ndarray, models: list[torch.nn.Module], featureFunctions: list[Callable]): 

    frame = Image

    if frame is None: 
        cv.destroyAllWindows()
        return
    
    cv.imshow('frame', frame)

    output = solveSudokuInImage(frame, models, featureFunctions)

    if(output is not None): 
        cv.imshow('output', output)

    cv.waitKey(5000)
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

    img_path = "data/sudokus"
    image = loadSudokus(img_path)[0][46]
    testImage(image, MODELS, FEATURE_FUNCTIONS)
    # testImage('C:\dev\Sudoku-Solver\data\sudokus\plain\image202.jpg', MODEL, FEATURE_FUNCTION)
    # testVideo('C:\dev\Sudoku-Solver\demo.mp4', MODEL, FEATURE_FUNCTION)
    pass