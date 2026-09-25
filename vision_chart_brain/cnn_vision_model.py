"""
2D Deep Convolutional Neural Network (CNN) for Candlestick Visual Pattern Recognition
Ingests [Batch, 3, 128, 128] RGB image tensors and outputs Softmax action probabilities:
  - P(0): HOLD / CHOP
  - P(1): BUY CALL (Bullish Visual Pattern)
  - P(2): BUY PUT (Bearish Visual Pattern)
"""

import os
from typing import Tuple, Dict
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


if TORCH_AVAILABLE:
    class CandlestickCNN(nn.Module):
        def __init__(self, num_classes: int = 3):
            super(CandlestickCNN, self).__init__()
            
            # Convolutional Feature Extractor
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.pool1 = nn.MaxPool2d(2, 2) # 128 -> 64

            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool2 = nn.MaxPool2d(2, 2) # 64 -> 32

            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.pool3 = nn.MaxPool2d(2, 2) # 32 -> 16

            self.global_pool = nn.AdaptiveAvgPool2d((4, 4)) # 128 x 4 x 4 = 2048

            # Dense Classification Head
            self.fc1 = nn.Linear(128 * 4 * 4, 128)
            self.dropout = nn.Dropout(0.35)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            
            x = self.global_pool(x)
            x = x.view(x.size(0), -1) # Flatten
            
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            logits = self.fc2(x)
            return logits

        def predict_probabilities(self, x: torch.Tensor) -> np.ndarray:
            self.eval()
            with torch.no_grad():
                logits = self.forward(x)
                probs = F.softmax(logits, dim=-1)
                return probs.cpu().numpy()
else:
    class CandlestickCNN:
        def __init__(self, num_classes: int = 3):
            pass
