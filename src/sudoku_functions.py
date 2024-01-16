import numpy as np
import os
import cv2 as cv
import shutil

def convertSudokus(src: str) -> np.ndarray: 

    filesInDirectory = os.listdir(f"{src}/plain")

    data = []
    targets  = []

    
    print("(+) Removing existing folders...")
    if os.path.exists(f"{src}/data"): 
        shutil.rmtree(f"{src}/data")

    print("(+) Creating data folder...")
    os.mkdir(f"{src}/data")

    print("(+) Converting sudoku image and dat data...")
    for file in filesInDirectory: 

        if not file.endswith(".jpg"): 
            continue
        
        image = cv.imread(f"{src}/plain/{file}")
        dat = open(f"{src}/plain/{file.replace('.jpg', '')}.dat", "r")
        lines = dat.readlines()[2:11]
        sudoku = [[int(digit) for digit in line.replace(" ", "").replace("\n", "")] for line in lines]

        data.append(image)
        targets.append(sudoku)

    data = np.array(data, dtype=object)
    targets = np.array(targets, dtype=np.uint8)

    print("(+) Saving data as .npy file...")
    np.save(f"{src}/data/data.npy", data)
    np.save(f"{src}/data/targets.npy", targets)

    print("(+) Sudoku conversion finished")
    return (data, targets)

def loadSudokus(src: str) -> np.ndarray:

    print("(+) Reading data from .npy file...")
    data = np.load(f"{src}/data/data.npy", allow_pickle=True)
    targets = np.load(f"{src}/data/targets.npy", allow_pickle=True)

    print("(+) Sudoku loading finished")
    return (data, targets)