from torch.utils.data import Dataset
import numpy as np
import torch

class ChessSPDataset(Dataset):
    """ Dataset for MCTS Self-Play data (positions, policy distributions, values) """
    def __init__(self, positions, policy_targets, values):
        self.positions = np.asarray(positions)
        self.policy_targets = np.asarray(policy_targets, dtype=np.float32)

        # Ensure values are float32 and shape (N, 1)
        if isinstance(values, np.ndarray) and values.ndim == 1:
            self.values = values.reshape(-1, 1).astype(np.float32)
        else:
            self.values = np.asarray(values, dtype=np.float32)
        if self.values.ndim != 2 or self.values.shape[1] != 1:
             try:
                  self.values = self.values.reshape(-1, 1)
             except Exception as e:
                  raise ValueError(f"Values shape issue. Expected (-1, 1), got {self.values.shape}. Error: {e}")

        # Validation
        n_samples = self.positions.shape[0]
        if not (self.policy_targets.shape[0] == n_samples and self.values.shape[0] == n_samples):
             raise ValueError("Mismatch in number of samples between positions, policies, and values.")
        if len(self.policy_targets.shape) != 2:
             raise ValueError(f"Policy targets should be 2D (N, num_moves), got {self.policy_targets.shape}")


    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        # Ensure positions are float32
        pos = self.positions[idx]
        if pos.dtype != np.float32:
            pos = pos.astype(np.float32)

        # Return tensors required by MCTSChessNN training_step
        return (
            torch.from_numpy(pos),
            torch.from_numpy(self.policy_targets[idx]), # Float tensor distribution
            torch.from_numpy(self.values[idx]) # Float tensor value shape (1,)
        )