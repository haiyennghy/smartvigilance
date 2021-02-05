import numpy as np
from torch.utils.data import Dataset

"""Check if we need it anymore"""
class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx] # Single Embedding Vector, it's Label