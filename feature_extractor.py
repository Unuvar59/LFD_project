"""
Feature extraction module using ResNet8
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from model_utils import DeepSVDDModel

class FeatureExtractor:
    def __init__(self, device=None):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = "./models/resnet8_feature_extractor.pth"
    
    def create_model(self):
        """Create ResNet8 model for feature extraction"""
        self.model = DeepSVDDModel(device=self.device)
        return self.model
    
    def save_model(self):
        """Save the feature extraction model"""
        if self.model is None:
            raise ValueError("Model not created yet. Call create_model() first.")
        
        os.makedirs("./models", exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Feature extraction model saved to {self.model_path}")
    
    def load_model(self):
        """Load the saved feature extraction model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.model = DeepSVDDModel(device=self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        print(f"Feature extraction model loaded from {self.model_path}")
        return self.model
    
    def extract_features(self, data_loader):
        """Extract features from data using ResNet8"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or create_model() first.")
        
        self.model.eval()
        features = []
        
        with torch.no_grad():
            for data, _ in data_loader:
                if self.device is not None:
                    data = data.to(self.device)
                
                # Get features from ResNet8 layers
                x = self.model.conv1(data)
                x = self.model.bn1(x)
                x = self.model.relu(x)
                
                x = self.model.layer1(x)
                x = self.model.layer2(x)
                x = self.model.layer3(x)
                
                x = self.model.avgpool(x)
                extracted_features = x.view(x.size(0), -1)
                features.append(extracted_features.cpu().numpy())
        
        return np.concatenate(features)

def create_and_save_feature_extractor(device=None):
    """Create and save ResNet8 feature extractor"""
    extractor = FeatureExtractor(device=device)
    extractor.create_model()
    extractor.save_model()
    return extractor

def load_feature_extractor(device=None):
    """Load saved ResNet8 feature extractor"""
    extractor = FeatureExtractor(device=device)
    extractor.load_model()
    return extractor 