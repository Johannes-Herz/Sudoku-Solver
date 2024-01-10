from src.sudoku_functions import convertSudokus, loadSudokus

SRC = "C:\dev\Sudoku-Solver\data\sudokus"

if __name__ == '__main__':
    data = convertSudokus(SRC)
    print(data[0].shape)
    print(data[1].shape)

    data = loadSudokus(SRC)
    print(data[0].shape)
    print(data[1].shape)