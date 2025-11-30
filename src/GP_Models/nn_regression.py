#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Network Regression for Superconductivity Prediction
===========================================================

This module implements a convolutional neural network (CNN) for predicting
superconducting critical temperatures (Tc) from histogram features and
symmetry descriptors.

The architecture combines:
- 1D convolutions over histogram bin features
- Fully connected layers for combined histogram + symmetry features
- MSE loss for regression training

Key Components
--------------
HistogramDataset : torch.utils.data.Dataset
    Custom dataset class for histogram features, symmetry vectors, and targets.
    
HistogramNN : torch.nn.Module
    CNN architecture with:
    - Conv1D layers for processing histogram bins
    - Fully connected layers for combined feature processing
    - Support for symmetry feature concatenation

train_epoch : function
    Training loop for one epoch with loss and R² computation.
    
evaluate : function
    Evaluation function for computing validation/test metrics.

Notes
-----
- The script includes data loading and preprocessing for histogram features
- Supports optional log-transformation of target values
- Includes visualization of training/test R² over epochs
- Designed for materials science applications (e.g., ICSD database)

Author
------
Omri Lesser

Examples
--------
To train the model, run the script directly:

    python nn_regression.py

The script will:
1. Load data from a pickle file
2. Split into train/test sets
3. Train the CNN for n_epochs
4. Plot R² scores over training
"""



import numpy as np
from matplotlib import pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Project imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "sc_train_gp-main"))

#%% Read data
with open(os.path.join(BASE_DIR, "sc_train_gp-main/data/regression_data_histogram&symmetry.pkl"), 'rb') as f:
    loaded_data = pickle.load(f)
    X = loaded_data['histogram_features']
    y = loaded_data['Tc']

X_non0 = []
y_non0 = []
BCS = ['Other','Chevrel','Oxide','Carbon']

for i in range(len(X)):
        X_non0.append(X[i])
        y_non0.append(y[i])

X_non0 = np.array(X_non0)
y_non0 = np.array(y_non0)

if 1: # Only 2nd order
    feat_names = loaded_data['histogram_feat_names']
    keep_indices = [index for index, string in enumerate(feat_names) if '2_ord' in string]
    X_non0 = X_non0[:, keep_indices, :, :]
    
    
if 0: # Only 1st + 2nd order
    feat_names = loaded_data['histogram_feat_names']
    keep_indices = [index for index, string in enumerate(feat_names) if ('2_ord' in string or '1_ord' in string)]
    X_non0 = X_non0[:, keep_indices, :, :]

y_clean = y_non0

X_flat = X_non0.reshape((X_non0.shape[0], np.prod(X_non0.shape[1:])))

symmetry_vector = np.array(loaded_data['symmetry_features'])

    
#%%
class HistogramDataset(Dataset):
    def __init__(self, X, symmetry, y):
        self.X = torch.FloatTensor(X)
        self.symmetry = torch.FloatTensor(symmetry)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.symmetry[idx], self.y[idx]

class HistogramNN(nn.Module):
    def __init__(self, n_properties, n_bins, n_symm):
        super().__init__()
        
        f = 1
        
        n1 = int(64 / f)
        n2 = int(150 / f)
        n3 = int(32 / f)
        
        # First process each property separately
        self.property_conv = nn.Sequential(
            nn.Conv1d(n_bins, n1, kernel_size=3, padding=1),
            nn.BatchNorm1d(n1),
            nn.ReLU(),
        )
        
        # Combined processing - input size now includes symmetry features
        combined_input_size = n1 * n_properties + n_symm
        
        self.combined = nn.Sequential(
            nn.Linear(combined_input_size, n2),
            nn.ReLU(),
            nn.Linear(n2, n3),
            nn.ReLU(),
            nn.Linear(n3, 1)
        )
        
    def forward(self, x, symmetry):
        batch_size = x.shape[0]
        
        # Split into centers and counts
        centers = x[..., 0]
        counts = x[..., 1]
        
        # Combine centers and counts through multiplication
        x = centers * counts
        
        # Process each property
        x = x.transpose(1, 2)
        x = self.property_conv(x)
        
        # Flatten histogram features
        x = x.reshape(batch_size, -1)
        
        # Concatenate with symmetry features
        x = torch.cat([x, symmetry], dim=1)
        
        # Process combined features
        x = self.combined(x)
        
        return x

def train_epoch(model, train_loader, criterion, optimizer, device, use_log=False):
    model.train()
    total_loss = 0
    all_predictions = []
    all_true_values = []
    
    for X, symmetry, y in train_loader:  # Modified to unpack symmetry
        X, symmetry, y = X.to(device), symmetry.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X, symmetry)  # Pass symmetry to model
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_predictions.append(outputs.detach().cpu().numpy())
        all_true_values.append(y.cpu().numpy())
    
    func = lambda x: np.exp(x) if use_log else x
    all_predictions = func(np.concatenate(all_predictions))
    all_true_values = func(np.concatenate(all_true_values))
    r2 = r2_score(all_true_values, all_predictions)
    
    return total_loss / len(train_loader), r2

def evaluate(model, data_loader, criterion, device, use_log=False):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_true_values = []
    
    with torch.no_grad():
        for X, symmetry, y in data_loader:  # Modified to unpack symmetry
            X, symmetry, y = X.to(device), symmetry.to(device), y.to(device)
            outputs = model(X, symmetry)  # Pass symmetry to model
            loss = criterion(outputs, y)
            total_loss += loss.item()
            all_predictions.append(outputs.cpu().numpy())
            all_true_values.append(y.cpu().numpy())
    
    func = lambda x: np.exp(x) if use_log else x
    all_predictions = func(np.concatenate(all_predictions))
    all_true_values = func(np.concatenate(all_true_values))
    r2 = r2_score(all_true_values, all_predictions)
    
    return total_loss / len(data_loader), r2


# Parameters
batch_size = 128 # 32
n_epochs = 4000
learning_rate = 2e-5
device = torch.device("mps")
use_log = False

# Split data - make sure to split symmetry_vector as well
X_train, X_test, symmetry_train, symmetry_test, y_train, y_test = train_test_split(
    X_non0, symmetry_vector, np.log(y_clean) if use_log else y_clean, test_size=0.2)

# Create data loaders with symmetry data
train_dataset = HistogramDataset(X_train, symmetry_train, y_train)
test_dataset = HistogramDataset(X_test, symmetry_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize model with symmetry dimension
n_properties = X_non0.shape[1]
n_bins = X_non0.shape[2]
n_symm = symmetry_vector.shape[1]
print(f"n_properties: {n_properties}, n_bins: {n_bins}, n_symm: {n_symm}")

model = HistogramNN(n_properties, n_bins, n_symm).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training history
train_r2_history = []
test_r2_history = []

# Training loop
for epoch in range(n_epochs):
    train_loss, train_r2 = train_epoch(model, train_loader, criterion, optimizer, device, use_log=use_log)
    test_loss, test_r2 = evaluate(model, test_loader, criterion, device, use_log=use_log)
        
    train_r2_history.append(train_r2)
    test_r2_history.append(test_r2)
    
    print(f'Epoch [{epoch+1}/{n_epochs}], Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}')


# Plot results
plt.figure(dpi=300)
plt.plot(train_r2_history, label='Train R²')
plt.plot(test_r2_history, label='Test R²')
plt.xlabel('Epoch')
plt.ylabel('R² Score')
plt.ylim([0.5, 1])
plt.title('Final R² - Train: %.03f, Test: %.03f' % (
    train_r2_history[-1], test_r2_history[-1]))
plt.legend()
plt.grid(True)

# Save results
# timestamp = datetime.now().strftime('%d%m%y_%H%M%S')
# base_filename = 'trained_models/model_icsd_%s' % timestamp
# plt.savefig(base_filename + '.png', bbox_inches='tight', pad_inches=0.05)
# torch.save(model, base_filename + '.pth')

plt.show()
