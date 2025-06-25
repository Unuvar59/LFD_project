"""
Utility functions and classes for Deep SVDD model
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, auc

class CIFAR10Loader:
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        self.normal_class = 8  # Ship class
        
        # Enhanced data transforms
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    def get_dataloaders(self):
        # Load CIFAR10
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=self.transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                   download=True, transform=self.test_transform)
        
        # Split data into normal and anomaly
        train_idx = torch.tensor(train_dataset.targets) == self.normal_class
        test_normal_idx = torch.tensor(test_dataset.targets) == self.normal_class
        test_anomaly_idx = torch.tensor(test_dataset.targets) != self.normal_class
        
        train_dataset.data = train_dataset.data[train_idx]
        train_dataset.targets = np.array(train_dataset.targets)[train_idx]
        
        test_normal_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                          download=True, transform=self.test_transform)
        test_normal_dataset.data = test_dataset.data[test_normal_idx]
        test_normal_dataset.targets = np.array(test_dataset.targets)[test_normal_idx]
        
        test_anomaly_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                           download=True, transform=self.test_transform)
        test_anomaly_dataset.data = test_dataset.data[test_anomaly_idx]
        test_anomaly_dataset.targets = np.array(test_dataset.targets)[test_anomaly_idx]
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_normal_loader = DataLoader(test_normal_dataset, batch_size=self.batch_size, shuffle=False)
        test_anomaly_loader = DataLoader(test_anomaly_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_normal_loader, test_anomaly_loader

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class DeepSVDDModel(nn.Module):
    def __init__(self, device):
        super(DeepSVDDModel, self).__init__()
        
        # Feature extractor
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(64, 128, stride=2)
        self.layer2 = self._make_layer(128, 256, stride=2)
        self.layer3 = self._make_layer(256, 512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 512
        
        # Deep SVDD parameters
        self.center = torch.zeros(self.feature_dim, device=device)
        self.radius = nn.Parameter(torch.tensor(0.0, device=device))
        
        self._initialize_weights()
        self.to(device)
        
        # For progressive training
        self.layers = [self.layer1, self.layer2, self.layer3]
        self.current_layer = 0
    
    def _make_layer(self, in_channels, out_channels, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        return BasicBlock(in_channels, out_channels, stride, downsample)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def set_layer_training(self, layer_idx):
        """Enable training for specific layer only"""
        for i, layer in enumerate(self.layers):
            for param in layer.parameters():
                param.requires_grad = (i == layer_idx)
        self.current_layer = layer_idx
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        features = x.view(x.size(0), -1)
        
        return features

def train_model(model, train_loader, epochs, device, lr=0.001, weight_decay=1e-4):
    # Progressive training schedule
    epochs_per_layer = epochs // 4  # 3 layers + 1 full training
    
    train_losses = []
    optimizer = None
    
    # Train each layer progressively
    for phase in range(4):
        if phase < 3:
            model.set_layer_training(phase)
            print(f"\nTraining layer {phase + 1}")
        else:
            # Final phase: train all layers
            for layer in model.layers:
                for param in layer.parameters():
                    param.requires_grad = True
            print("\nFinal phase: training all layers")
        
        # Create new optimizer for each phase with cosine annealing learning rate
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr * (0.3 ** phase),  # Reduce learning rate more gradually
            weight_decay=weight_decay * (10 ** phase),  # Increase weight decay for later phases
            betas=(0.9, 0.999)  # Default Adam betas
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs_per_layer,
            eta_min=1e-6
        )
        
        # Initialize/update center
        features_list = []
        model.eval()
        with torch.no_grad():
            for data, _ in train_loader:
                features = model(data.to(device))
                features_list.append(features)
        features = torch.cat(features_list, dim=0)
        if phase == 0:
            model.center = torch.mean(features, dim=0)
        else:
            # Update center with momentum (higher momentum for stability)
            new_center = torch.mean(features, dim=0)
            model.center = 0.99 * model.center + 0.01 * new_center
        
        # Train for this phase
        model.train()
        for epoch in range(epochs_per_layer):
            epoch_loss = 0
            for data, _ in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                
                features = model(data)
                dist = torch.sum((features - model.center) ** 2, dim=1)
                loss = torch.mean(dist)
                
                if phase == 3:  # Add radius regularization in final phase
                    loss += 0.005 * model.radius  # Reduced radius regularization
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Step the learning rate scheduler
            scheduler.step()
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 5 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f'Phase {phase + 1}, Epoch [{epoch+1}/{epochs_per_layer}], '
                      f'Loss: {avg_loss:.4f}, LR: {current_lr:.6f}')
    
    return train_losses

def evaluate_model(model, test_normal_loader, test_anomaly_loader, device):
    model.eval()
    normal_scores = []
    anomaly_scores = []
    
    with torch.no_grad():
        for data, _ in test_normal_loader:
            features = model(data.to(device))
            scores = torch.sum((features - model.center) ** 2, dim=1)
            normal_scores.extend(scores.cpu().numpy())
        
        for data, _ in test_anomaly_loader:
            features = model(data.to(device))
            scores = torch.sum((features - model.center) ** 2, dim=1)
            anomaly_scores.extend(scores.cpu().numpy())
    
    # Calculate ROC curve and AUC score
    labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])
    scores = np.concatenate([normal_scores, anomaly_scores])
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    return {
        'normal_scores': normal_scores,
        'anomaly_scores': anomaly_scores,
        'fpr': fpr,
        'tpr': tpr,
        'auc_score': roc_auc
    }

def train_traditional_methods(train_loader, test_normal_loader, test_anomaly_loader, feature_extractor=None, device=None):
    """
    Train and evaluate traditional anomaly detection methods using ResNet8 features
    Args:
        train_loader: DataLoader for training data
        test_normal_loader: DataLoader for normal test data
        test_anomaly_loader: DataLoader for anomaly test data
        feature_extractor: FeatureExtractor instance for feature extraction
        device: torch device (cuda/cpu)
    """
    # Extract features using ResNet8 backbone
    def get_features(loader):
        if feature_extractor is not None:
            return feature_extractor.extract_features(loader)
        else:
            # Fallback to raw data if no feature extractor provided
            features = []
            for data, _ in loader:
                features.append(data.view(data.size(0), -1).numpy())
            return np.concatenate(features)
    
    train_features = get_features(train_loader)
    test_normal_features = get_features(test_normal_loader)
    test_anomaly_features = get_features(test_anomaly_loader)
    
    # Standardize features
    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0)
    train_features = (train_features - mean) / (std + 1e-10)
    test_normal_features = (test_normal_features - mean) / (std + 1e-10)
    test_anomaly_features = (test_anomaly_features - mean) / (std + 1e-10)
    
    # Train and evaluate One-Class SVM with adjusted parameters
    print("\nTraining One-Class SVM...")
    ocsvm = OneClassSVM(
        kernel='rbf',  # RBF kernel for ResNet features
        nu=0.1,  # Moderate nu value
        gamma='scale'
    )
    ocsvm.fit(train_features)
    
    normal_scores_svm = -ocsvm.score_samples(test_normal_features)
    anomaly_scores_svm = -ocsvm.score_samples(test_anomaly_features)
    
    labels_svm = np.concatenate([np.zeros(len(normal_scores_svm)), np.ones(len(anomaly_scores_svm))])
    scores_svm = np.concatenate([normal_scores_svm, anomaly_scores_svm])
    fpr_svm, tpr_svm, _ = roc_curve(labels_svm, scores_svm)
    roc_auc_svm = auc(fpr_svm, tpr_svm)
    
    # Train and evaluate Isolation Forest with adjusted parameters
    print("Training Isolation Forest...")
    iforest = IsolationForest(
        n_estimators=100,  # Moderate number of trees
        max_samples='auto',
        contamination=0.1,  # Moderate contamination
        max_features=1.0,  # Use all features since they're already processed by ResNet
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    iforest.fit(train_features)
    
    normal_scores_if = -iforest.score_samples(test_normal_features)
    anomaly_scores_if = -iforest.score_samples(test_anomaly_features)
    
    labels_if = np.concatenate([np.zeros(len(normal_scores_if)), np.ones(len(anomaly_scores_if))])
    scores_if = np.concatenate([normal_scores_if, anomaly_scores_if])
    fpr_if, tpr_if, _ = roc_curve(labels_if, scores_if)
    roc_auc_if = auc(fpr_if, tpr_if)
    
    return {
        'oc_svm': {
            'normal_scores': normal_scores_svm,
            'anomaly_scores': anomaly_scores_svm,
            'fpr': fpr_svm,
            'tpr': tpr_svm,
            'auc_score': roc_auc_svm
        },
        'isolation_forest': {
            'normal_scores': normal_scores_if,
            'anomaly_scores': anomaly_scores_if,
            'fpr': fpr_if,
            'tpr': tpr_if,
            'auc_score': roc_auc_if
        }
    } 