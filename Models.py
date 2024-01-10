
import torch

class PixelModel(torch.nn.Module): 
    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(1600, 240, dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(240, 80, dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(80, 9, dtype=torch.float32),
        )
    def forward(self, a):
        return self.mlp(a)

class PixelDensityModel(torch.nn.Module): 
    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(80, 240, dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(240, 120, dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 80, dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(80, 9, dtype=torch.float32),
        )
    def forward(self, a):
        return self.mlp(a)
    
class ImageGradientDensityModel(torch.nn.Module): 
    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(8, 60, dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(60, 40, dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(40, 20, dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 9, dtype=torch.float32),
        )
    def forward(self, a):
        return self.mlp(a)

class ConvolutionalModel(torch.nn.Module): 
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, (5, 5), (1, 1), (0, 0), dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, (5, 5), (1, 1), (0, 0), dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, (5, 5), (1, 1), (0, 0), dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, (5, 5), (1, 1), (0, 0), dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, (5, 5), (1, 1), (0, 0), dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, (5, 5), (1, 1), (0, 0), dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, (5, 5), (1, 1), (0, 0), dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, (5, 5), (1, 1), (0, 0), dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, (5, 5), (1, 1), (0, 0), dtype=torch.float32),
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(256, 32, dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16, dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 9, dtype=torch.float32),
        )
    def forward(self, a):
        a = self.conv(a)
        a = torch.flatten(a, start_dim=1)
        return self.mlp(a)
