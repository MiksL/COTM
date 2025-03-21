from torch.utils.data import Dataset
import numpy as np
import torch

class ChessDataset(Dataset):
    """Custom chess position dataset using pre-encoded data"""
    def __init__(self, positions, moves, values):
        self.positions = positions
        
        # Ensure moves are in the right format
        self.moves = np.asarray(moves, dtype=np.int32)
        
        # Format check for values - should be a 2D array with the shape (N, 1)
        if isinstance(values, np.ndarray) and values.ndim == 1:
            # Convert 1D array to 2D
            self.values = values.reshape(-1, 1)
        else:
            # Make sure it's a numpy array
            self.values = np.asarray(values, dtype=np.float32)
            if self.values.ndim == 2 and self.values.shape[1] != 1:
                # Reshape if it's not the correct shape
                self.values = self.values.reshape(-1, 1)
        
    def __len__(self):
        return len(self.moves)
        
    def __getitem__(self, idx):
        # Convert position to float32 if not already
        if self.positions.dtype != np.float32:
            pos = self.positions[idx].astype(np.float32)
        else:
            pos = self.positions[idx]
        
        # Prepare data for batch
        return (
            torch.from_numpy(pos).float(),
            torch.tensor(self.moves[idx], dtype=torch.long),
            torch.from_numpy(self.values[idx]).float()
        )