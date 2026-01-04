# src/feature_extractor.py
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class FeatureExtractor:
    def __init__(self):
        # 1. Thiết lập thiết bị (GPU nếu có, ngược lại là CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model.to(self.device)
        self.model.eval()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),  # Chuyển từ numpy/PIL (0-255) sang Tensor (0-1) và đổi chiều (HWC -> CHW)
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def apply_clahe(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = self.clahe.apply(l)
        result = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    def extract(self, img):
        if img is None: return None
        img = cv2.resize(img, (224, 224))
        img = self.apply_clahe(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = self.preprocess(img)
        input_batch = input_tensor.unsqueeze(0)
        input_batch = input_batch.to(self.device)
        with torch.no_grad():
            features = self.model(input_batch)
        return features.cpu().numpy().flatten()