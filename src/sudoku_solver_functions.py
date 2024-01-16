import torch
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
from collections.abc import Callable


'''
    Sudoku Detection
'''

def applyInitialThreshold(image: np.ndarray) -> np.ndarray: 
    
    blurred = cv.GaussianBlur(image, (7, 7), 3)

    thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    thresh = cv.bitwise_not(thresh)
    _, thresh = cv.threshold(thresh, 128, 255, cv.THRESH_BINARY_INV)

    return thresh.astype(np.uint8)

def checkContoursForSize(image: np.ndarray, contours: list) -> list: 

    inputImageHeightThreshold = [image.shape[0] * 0.95, image.shape[0]]
    inputImageWidthThreshold = [image.shape[1] * 0.95, image.shape[1]]
    minRatioToImage = float(float(1)/float(3))

    contourSelection = []

    for contour in contours: 
        x, y, w, h = cv.boundingRect(contour)
        if w < inputImageWidthThreshold[1] * minRatioToImage or h < inputImageHeightThreshold[1] * minRatioToImage:
            continue
        if w > inputImageWidthThreshold[0] or h > inputImageHeightThreshold[0]: 
            continue
        contourSelection.append(contour)

    return contourSelection

def checkContoursForSquares(contours: list) -> list:

    contourAspectRatioThreshold = [0.9, 1.1]

    contourSelection = []

    for contour in contours: 
        approximation = cv.approxPolyDP(contour, 0.02 * cv.arcLength(contour, True), True)
        if len(approximation) == 4:
            x, y, w, h = cv.boundingRect(contour)
            ratio = float(w)/float(h)
            if ratio > contourAspectRatioThreshold[0] and ratio < contourAspectRatioThreshold[1]: 
                contourSelection.append(approximation)

    return contourSelection

def detectSudokuInGrayScaleImage(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]: 

    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv.cvtColor(image.copy(), cv.COLOR_BGR2GRAY) 

    thresh = applyInitialThreshold(image.copy())

    contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    contours = checkContoursForSize(image.copy(), contours)
    contours = checkContoursForSquares(contours)

    if len(contours) == 0: 
        return None, None
    elif len(contours) > 1: 
        contours = sorted(contours, key=cv.contourArea, reverse=True)

    sudokuContour = contours[0]

    sudokuContour = sudokuContour.reshape(4, 2)
    sudokuImageGrayScale = four_point_transform(image, sudokuContour)
    sudokuImageGrayScale = cv.resize(sudokuImageGrayScale, (900, 900))

    return sudokuImageGrayScale.astype(np.uint8), sudokuContour

'''
    Cell Extraction
'''

def retrieveCellsFromSudokuImage(sudokuImage: np.ndarray)  -> np.ndarray: 
    
    if sudokuImage.shape[0] != 900 or sudokuImage.shape[1]!= 900:
        return None
    
    cellSize = int(sudokuImage.shape[0] / 9)
    cells = []

    for x in range(0, 9): 
        cells.append([])
        for y in range(0, 9): 
            cell = sudokuImage[x * cellSize: x * cellSize + cellSize, y * cellSize: y * cellSize + cellSize]
            cell = cell.astype(np.uint8)
            cells[x].append(cell)

    return np.array(cells)

'''
    Cell Classification
'''

def preprocessCellImage(cellImage: np.ndarray): 

    thresh = cv.threshold(cellImage, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return False, None
    
    contour = max(contours, key=cv.contourArea)

    (h, w) = thresh.shape

    _, _, w_contour, h_contour = cv.boundingRect(contour)

    if w_contour*h_contour < 0.04*h*w:
        return False, None

    mask = np.zeros(thresh.shape, dtype=np.uint8)
    cv.drawContours(mask, [contour], -1, 255, cv.FILLED)

    (h, w) = thresh.shape
    percentageFilled = cv.countNonZero(mask) / float(w *h)

    if percentageFilled < 0.02:
        return False, None
    
    cellImage = cv.bitwise_and(thresh, thresh, mask=mask)
    return True, cellImage.astype(np.uint8)

def classifyCells(cells: np.ndarray, models: list[torch.nn.Module]|torch.nn.Module, featureFunctions: list[Callable]|Callable): 

    if(not isinstance(models, list)):
        models = [models]

    if(not isinstance(featureFunctions, list)):
        featureFunctions = [featureFunctions]

    sudoku = np.zeros((9, 9))

    for xIndex, row in enumerate(cells): 
        for yIndex, cell in enumerate(row):
            cell = cv.resize(cell, (40, 40))
            needsClassification, cell = preprocessCellImage(cell)

            if not needsClassification: 
                continue

            cell = ((255 - cell) / 255).astype(np.uint8)

            digitlist = []

            for model, featureFunction in zip(models, featureFunctions):

                X: np.ndarray = featureFunction(cell)
                X = torch.tensor(X, dtype=torch.float32)
                y = model(X)
                y = y.detach().cpu()
                digitlist.append(torch.argmax(y, 0).item() + 1)

            sudoku[xIndex, yIndex]:int = voteForDigit(digitlist)

    return np.array(sudoku, dtype=np.uint8)

def voteForDigit(digitlist: list[int]):

    counts = {num: digitlist.count(num) for num in set(digitlist)} 

    freq = {val:[i for i in counts.keys() if counts[i] == val] for val in counts.values()}

    max_votes = freq[max(freq.keys())]

    positions = {digitlist.index(num): num for num in max_votes}

    return positions[min(positions.keys())]

'''
    Sudoku Solving
'''

def solveSudoku(sudoku: np.ndarray) -> tuple[bool, np.ndarray]:

    for x in range(sudoku.shape[0]):

        for y in range(sudoku.shape[1]):
            if sudoku[x][y] == 0:
                possibleDigits: list[int] = [digit for digit in range(1, 10) if digit not in digitsInSudokuLines(sudoku, x, y) if digit not in digitsInSudokuBox(sudoku, x, y)]
                
                for digit in possibleDigits: 
                    sudoku[x][y] = digit
                    if solveSudoku(sudoku)[0]:
                        return True, sudoku
                    sudoku[x][y] = 0

                return False, sudoku
            
    return True, sudoku

def digitsInSudokuBox(sudoku: np.ndarray, x: int, y: int) -> np.ndarray:

    _x = int(x / 3) * 3
    _y = int(y / 3) * 3
    box = sudoku[_x : _x + 3, _y : _y + 3]
    box[x % 3, y % 3] = 0
    digitsInBox = np.unique(box)

    return digitsInBox

def digitsInSudokuLines(sudoku: np.ndarray, x: int, y: int) -> np.ndarray:

    horizontalLine = sudoku[x,:]
    horizontalLine[y] = 0
    digitsInHorizontalLine = np.unique(horizontalLine)
    verticalLine = sudoku[:,y]
    verticalLine[x] = 0
    digitsInVerticalLine = np.unique(verticalLine)
    digitsInLines = np.unique(np.concatenate((digitsInHorizontalLine, digitsInVerticalLine)))

    return digitsInLines

'''
    Sudoku Image Generation
'''

def createSudokuImageFromSudoku(sudoku: np.ndarray) -> np.ndarray:

    sudokuImage = np.ones((900, 900),) * 255
    sudokuImage = sudokuImage.astype(np.uint8)

    for x in range(0, 9):
        for y in range(0, 9):
            if sudoku[x, y] == 0:
                continue
            text = str(sudoku[x, y])
            textSize = cv.getTextSize(text, cv.FONT_HERSHEY_PLAIN, 3, 2)[0]
            sudokuImage = cv.putText(sudokuImage, text, (y * 100 + 50 - int(textSize[0] / 2), x * 100 + 50 + int(textSize[1] / 2)), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 4, lineType=cv.LINE_8)
    
    return sudokuImage.astype(np.uint8)

'''
    Sudoku Image Remapping
'''

def createReversedSudoku(image: np.ndarray, sudokuImage: np.ndarray, sudokuContours: np.ndarray) -> np.ndarray: 

    isRGB = False
    if len(image.shape) == 3 and image.shape[2] == 3:
        isRGB = True

    dstPointsSortedByNorm = [(np.linalg.norm(point), point) for point in sudokuContours]
    dstPointsSortedByNorm.sort(key = lambda item: item[0])
    dstPointsSortedByNorm = [item[1] for item in dstPointsSortedByNorm]
    if dstPointsSortedByNorm[1][0] > dstPointsSortedByNorm[1][1] and dstPointsSortedByNorm[2][0] <= dstPointsSortedByNorm[2][1]:
        buff = dstPointsSortedByNorm[1]
        dstPointsSortedByNorm[1] = dstPointsSortedByNorm[2]
        dstPointsSortedByNorm[2] = buff
    dstPointsSortedByNorm = np.array(dstPointsSortedByNorm)

    srcPoints = np.array([[[0, 0]], 
                            [[0, sudokuImage.shape[0]]], 
                            [[sudokuImage.shape[1], 0]],
                            [[sudokuImage.shape[1], sudokuImage.shape[0]]]], dtype=np.float32)
    dstPoints = dstPointsSortedByNorm.astype(np.float32)

    reversedSudoku = cv.warpPerspective(255 - sudokuImage, cv.getPerspectiveTransform(srcPoints, dstPoints), (image.shape[1], image.shape[0]))
    reversedSudoku = reversedSudoku.astype(np.uint8)
    _, reversedSudoku = cv.threshold(reversedSudoku, 128, 255, cv.THRESH_BINARY)
    if isRGB:
        reversedSudoku = np.array([reversedSudoku.copy(), reversedSudoku.copy(), reversedSudoku.copy()])
        reversedSudoku = np.transpose(reversedSudoku, (1, 2, 0))
    
    return reversedSudoku.astype(np.uint8)

def createSudokuMask(image: np.ndarray, sudokuContours: np.ndarray) -> np.ndarray:

    sudokuMask = np.ones_like(image) * 255
    cv.drawContours(sudokuMask, [sudokuContours], -1, (0, 0, 0), thickness=cv.FILLED)

    return sudokuMask.astype(np.uint8)

def mapSudokuImageOnImage(image: np.ndarray, sudokuImage: np.ndarray, sudokuContours: np.ndarray) -> np.ndarray: 

    reversedSudoku = createReversedSudoku(image, sudokuImage, sudokuContours)
    
    reversedSudokuInverted = 255 - reversedSudoku
    
    redNumbers = reversedSudoku.copy()
    redNumbers[:,:,0][redNumbers[:,:,0] == 255] = 0
    redNumbers[:,:,1][redNumbers[:,:,1] == 255] = 0
    
    maskedInput = redNumbers + image * ( reversedSudokuInverted / 255 )
    maskedInput = maskedInput.astype(np.uint8)

    return maskedInput
'''
    Sudoku Solver
'''

def solveSudokuInImage(image: np.ndarray, models: list[torch.nn.Module]|torch.nn.Module, featureFunctions: list[Callable]|Callable): 

    img = image.copy()
    sudokuImage, sudokuContours = detectSudokuInGrayScaleImage(img)

    cells = retrieveCellsFromSudokuImage(sudokuImage)

    sudoku = classifyCells(cells, models, featureFunctions)
    sudoku = sudoku.astype(np.uint8)

    isSolved, solvedSudoku = solveSudoku(sudoku.copy()) 

    if not isSolved: 
        return None

    mappingSudoku = np.zeros((9,9))
    mappingSudoku = np.where(sudoku == 0, solvedSudoku, mappingSudoku).astype(np.uint8)
    
    solvedSudokuImage = createSudokuImageFromSudoku(mappingSudoku)
    output = mapSudokuImageOnImage(img.copy(), solvedSudokuImage, sudokuContours)

    return output