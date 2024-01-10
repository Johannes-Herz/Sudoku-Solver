from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import shutil
import cv2 as cv

def generateDigits(dst: str, imagesPerDigit: int, fontPaths: list[str] = None, trainTestSplit: float = 0.8, applyRandomNoise: float = 0.0):

    TEXT_SPAWN_RADIUS = 5
    IMAGE_SIZE = 40
    MIN_FONT_SIZE = 24
    MAX_FONT_SIZE = 36

    print("(+) Removing existing folders...")
    if os.path.exists(f"{dst}/train"): 
        shutil.rmtree(f"{dst}/train")
    if os.path.exists(f"{dst}/test"): 
        shutil.rmtree(f"{dst}/test")

    print("(+) Creating train and test folders...")
    os.mkdir(f"{dst}/train")
    os.mkdir(f"{dst}/test")

    trainData = []
    trainTargets  = []
    testData  = []
    testTargets = []

    fontCount = len(fontPaths) if fontPaths is not None else None

    print("(+) Generating images...")
    for digit in range(1, 10):
        print(f"(+) Generating images for digit {digit}...")
        text = str(digit)

        for idx in range(imagesPerDigit):     
            fontPathIndex: int = np.random.randint(fontCount) if fontCount is not None else None
            fontPath: str = fontPaths[fontPathIndex] if fontPathIndex is not None else None
            
            x: int = np.random.randint(int(IMAGE_SIZE / 2) - TEXT_SPAWN_RADIUS, int(IMAGE_SIZE / 2) + TEXT_SPAWN_RADIUS)
            y: int = np.random.randint(int(IMAGE_SIZE / 2) - TEXT_SPAWN_RADIUS, int(IMAGE_SIZE / 2) + TEXT_SPAWN_RADIUS)

            fontSize: int = np.random.randint(MIN_FONT_SIZE, MAX_FONT_SIZE)

            result = generateImage(text, (IMAGE_SIZE, IMAGE_SIZE), fontPath, fontSize, (x, y))
            imageMatrix, image = result

            is_test: bool = int(imagesPerDigit * trainTestSplit) <= idx

            if applyRandomNoise > 0: 
                noise = np.array([-255 if np.random.uniform(0, 1) < applyRandomNoise else 0 for i in range(imageMatrix.shape[0] * imageMatrix.shape[1])])
                noise = noise.reshape((imageMatrix.shape[0], imageMatrix.shape[1]))
                imageMatrix = imageMatrix + noise
            cv.imwrite(f"{dst}/{'test' if is_test else 'train'}/{digit}_{idx + 1}.jpg", imageMatrix)

            if is_test: 
                testData.append(imageMatrix)
                testTargets.append(digit)
            else: 
                trainData.append(imageMatrix)
                trainTargets.append(digit)

    trainSet = (np.array(trainData, dtype=np.uint8), np.array(trainTargets, dtype=np.int8))
    testSet = (np.array(testData, dtype=np.uint8), np.array(testTargets, dtype=np.int8))

    print("(+) Saving data as .npy file...")
    np.save(f"{dst}/train/data.npy", trainSet[0])
    np.save(f"{dst}/train/targets.npy", trainSet[1])
    np.save(f"{dst}/test/data.npy", testSet[0])
    np.save(f"{dst}/test/targets.npy", testSet[1])

    print("(+) Digit generation finished")
    return (trainSet, testSet)

def generateImage(text: str, imageSize: tuple[int, int], fontPath: str, fontSize: int, pos: tuple[int, int]):

    image = Image.new("L",imageSize, color=255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(fontPath, fontSize)  if fontPath is not None else ImageFont.load_default(fontSize)

    text_bbox = draw.textbbox((0, 0), text, font=font)
    x = pos[0] - (text_bbox[2] - text_bbox[0]) // 2
    y = pos[1] - (text_bbox[3] - text_bbox[1]) // 2

    draw.text((x, y), text, fill=0, font=font)
    return (np.array(image, dtype=np.uint8), image)

def loadDigits(src: str):

    print("(+) Reading data from .npy file...")
    trainData = np.load(f"{src}/train/data.npy")
    trainTargets = np.load(f"{src}/train/targets.npy")
    testData = np.load(f"{src}/test/data.npy")
    testTargets = np.load(f"{src}/test/targets.npy")

    trainSet = (trainData, trainTargets)
    testSet = (testData, testTargets)

    print("(+) Digit loading finished")
    return (trainSet, testSet)

def loadTrainDigits(src: str):

    print("(+) Reading data from .npy file...")
    trainData = np.load(f"{src}/train/data.npy")
    trainTargets = np.load(f"{src}/train/targets.npy")

    trainSet = (trainData, trainTargets)

    print("(+) Digit loading finished")
    return trainSet

def loadTestDigits(src: str):

    print("(+) Reading data from .npy file...")
    testData = np.load(f"{src}/test/data.npy")
    testTargets = np.load(f"{src}/test/targets.npy")

    testSet = (testData, testTargets)

    print("(+) Digit loading finished")
    return testSet

def applyFunction(src: str, dst: str, function, dtype = np.uint8):

    print("(+) Removing existing folders...")
    if os.path.exists(f"{dst}/train"): 
        shutil.rmtree(f"{dst}/train")
        
    if os.path.exists(f"{dst}/test"): 
        shutil.rmtree(f"{dst}/test")
        
    print("(+) Creating train and test folders...")
    os.mkdir(f"{dst}/train")
    os.mkdir(f"{dst}/test")

    data = loadDigits(src)

    print("(+) Preprocessing data...")
    processedTrainData = np.array([function(sample) for sample in data[0][0]], dtype=dtype)
    processedTestData = np.array([function(sample) for sample in data[1][0]], dtype=dtype)

    print("(+) Saving data as .npy file...")
    np.save(f"{dst}/train/data.npy", processedTrainData)
    np.save(f"{dst}/train/targets.npy", data[0][1])
    np.save(f"{dst}/test/data.npy", processedTestData)
    np.save(f"{dst}/test/targets.npy", data[1][1])

    trainSet = (processedTrainData, data[0][1])
    testSet = (processedTestData, data[1][1])

    print("(+) Digit preprocessing finished")
    return (trainSet, testSet)