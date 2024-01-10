import numpy as np
import cv2 as cv

def pixelFeatureFunction(cellImage: np.ndarray) -> np.ndarray:
    return cellImage.flatten().astype(np.float32)

def pixelDensityFeatureFunction(cellImage: np.ndarray) -> np.ndarray:
    
    cellImage = 1 - cellImage

    xAxisDensity = np.sum(cellImage, axis=0) / cellImage.shape[1]
    yAxisDensity = np.sum(cellImage, axis=1) / cellImage.shape[0]

    pixelDensity = np.append(xAxisDensity, yAxisDensity)

    return pixelDensity.astype(np.float32)

def imageGradientDensityFeatureFunction(cellImage: np.ndarray) -> np.ndarray:
    
    gx = cv.Sobel(cellImage, ddepth=cv.CV_32F, dx=1, dy=0, ksize=3)
    gy = cv.Sobel(cellImage, ddepth=cv.CV_32F, dx=0, dy=1, ksize=3)

    coloredMap = np.zeros_like(gx, dtype=np.float32)
    coloredMap[np.logical_and(gx == 0, gy == 0)] = np.nan
    coloredMap[np.logical_and(gx > 0, gy > 0)] = 0
    coloredMap[np.logical_and(gx < 0, gy > 0)] = 1
    coloredMap[np.logical_and(gx < 0, gy < 0)] = 2
    coloredMap[np.logical_and(gx > 0, gy < 0)] = 3
    coloredMap[np.logical_and(gx == 0, gy > 0)] = 4
    coloredMap[np.logical_and(gx == 0, gy < 0)] = 5
    coloredMap[np.logical_and(gx > 0, gy == 0)] = 6
    coloredMap[np.logical_and(gx < 0, gy == 0)] = 7

    vals = [i for i in range(0, 8)]
    totalNonNaNPixels = np.count_nonzero(~np.isnan(coloredMap))
    unique, count = np.unique(coloredMap, return_counts=True)
    imageGradientDensity = np.array([float(count[np.where(unique == val)[0][0]] / totalNonNaNPixels) if val in unique else 0.0 for val in vals])
    
    return imageGradientDensity.astype(np.float32)