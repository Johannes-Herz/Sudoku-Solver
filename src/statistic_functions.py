from src.sudoku_solver_functions import *

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def evaluateModelOnGeneratedDigits(model: torch.nn.Module, digits: tuple[np.ndarray, np.ndarray]) -> None: 

    print(f'(+) Starting model evaluation...')
    X = torch.tensor(digits[0], dtype=torch.float32)

    total = digits[0].shape[0]

    correct = 0
    digitStatistics = {}
    for i in range(1, 10): 
        digitStatistics[i] = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0 }

    _Y = model(X)
    _Y: np.ndarray = torch.argmax(_Y, dim=1).detach().numpy()

    for _y, y in zip(_Y, digits[1]): 
        _y += 1
        if _y == y: 
            correct += 1
        digitStatistics[y][_y] = digitStatistics[y][_y] + 1

    print(f'(+) Model achieved an overall accuracy of {round(correct/total, 4)}!')

    fig, axs = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)
    fig.tight_layout()
    x = [str(i) for i in range(1, 10)]
    ys = [np.array([*statistic.values()]) / np.sum(np.array([*statistic.values()])) for statistic in digitStatistics.values()]
    for digit, (axis, y) in enumerate(zip(axs.flatten(), ys)):
        axis.set_xticks(np.arange(0, 1.1, 0.1))
        axis.set_title(str(digit + 1))
        axis.barh(x, y)
        for index, value in enumerate(y):
            axis.text( 0.5, index - 0.1, "{:.4f}".format(round(value, 4)), color='black', fontweight='light')

    print(f'(+) Model evaluation finished')

def evaluateModelOnSudokus(model: torch.nn.Module, featureFunction, sudokus: tuple[np.ndarray, np.ndarray]): 

    print(f'(+) Starting model evaluation...')
    totalSudokusFound = 0
    totalDigits = sudokus[0].shape[0] * 81

    correctDigits = 0
    solvedSudokus = 0
    digitStatistics = {}
    for i in range(0, 10): 
        digitStatistics[i] = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0 }

    for data, label in zip(sudokus[0], sudokus[1]):


        img = data.copy()
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sudokuImageGrayScale, sudokuContours = detectSudokuInGrayScaleImage(imgGray)

        if sudokuImageGrayScale is None or sudokuContours is None: 
            continue

        cells = retrieveCellsFromSudokuImage(sudokuImageGrayScale)

        sudoku = classifyCells(cells, model, featureFunction)
        totalSudokusFound += 1

        wouldHaveSolved = True
        for _y, y in zip(sudoku.flatten(), label.flatten()):
            if _y == y: 
                correctDigits += 1
            else: 
                wouldHaveSolved = False
            digitStatistics[y][_y] = digitStatistics[y][_y] + 1
        if wouldHaveSolved:
            solvedSudokus += 1

    print(f'(+) Model achieved an overall accuracy of {round(correctDigits/totalDigits, 4)}!')
    print(f'(+) Model would have solved {solvedSudokus} out of {totalSudokusFound}!')

    fig, axs = plt.subplots(4, 3, figsize=(12, 12), sharex=True, sharey=True)
    fig.tight_layout()
    x = [str(i) for i in range(0, 10)]
    ys = [np.array([*statistic.values()]) / np.sum(np.array([*statistic.values()])) for statistic in digitStatistics.values()]
    axs[0,0].axis('off')
    axs[0,2].axis('off')
    axs = [axis for index, axis in enumerate(axs.flatten()) if index != 0 and index != 2]
    for digit, (axis, y) in enumerate(zip(axs, ys)):
        axis.set_xticks(np.arange(0, 1.1, 0.1))
        axis.set_title(str(digit))
        barChart = axis.barh(x, y)
        for index, value in enumerate(y):
            axis.text( 0.5, index - 0.2, "{:.4f}".format(round(value, 4)), color='black', fontweight='light')

    print(f'(+) Model evaluation finished')