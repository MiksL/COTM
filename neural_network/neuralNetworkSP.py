import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from neural_network.baseNeuralNetwork import BaseChessNN # Import the base class

class MCTSChessNN(BaseChessNN):
    """
    Neural Network subclass specifically for training with MCTS-generated data.
    Overrides training/validation steps to handle policy distributions.
    """
    def __init__(self, input_channels=18, num_res_blocks=10, policy_output_size=1968, learning_rate=1e-3):
        super().__init__(input_channels, num_res_blocks, policy_output_size, learning_rate)

    def _calculate_loss(self, batch):
        """
        Calculates the combined loss for policy (distribution target) and value.
        """
        positions, policy_targets, value_targets = batch

        # Ensure correct types and device
        positions = positions.to(self.device).float()
        policy_targets = policy_targets.to(self.device).float()
        value_targets = value_targets.to(self.device).float()

        # Forward pass
        policy_logits, value_pred = self(positions)

        # --- Policy Loss ---
        # Use CrossEntropyLoss: it expects logits and probability distributions as targets.
        # It implicitly applies LogSoftmax to logits and computes NLLLoss.
        # Ensure policy_targets sums to 1 (it should from MCTS generation)
        # Add a small epsilon for numerical stability if needed, though usually handled by PyTorch.
        # policy_targets = policy_targets + 1e-8
        # policy_targets = policy_targets / policy_targets.sum(dim=1, keepdim=True)

        policy_loss = F.cross_entropy(policy_logits, policy_targets)

        # --- Value Loss ---
        # Use Mean Squared Error loss
        value_loss = F.mse_loss(value_pred, value_targets)

        # --- Combined Loss (AlphaZero often weights value loss less) ---
        # You might experiment with the weighting factor (e.g., 1.0, 0.5, 0.25)
        combined_loss = policy_loss + 1.0 * value_loss

        return combined_loss, policy_loss, value_loss, policy_logits, value_pred

    def training_step(self, batch, batch_idx):
        """
        Performs a training step using MCTS data (policy distributions).
        """
        total_loss, policy_loss, value_loss, _, _ = self._calculate_loss(batch)

        # Logging
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch[0].size(0))
        self.log('train_policy_loss', policy_loss, on_step=False, on_epoch=True, logger=True, batch_size=batch[0].size(0))
        self.log('train_value_loss', value_loss, on_step=False, on_epoch=True, logger=True, batch_size=batch[0].size(0))
        # Note: Simple 'accuracy' isn't directly applicable when the target is a distribution.
        # You could log top-1 accuracy (if predicted argmax matches target argmax)
        # or KL divergence if needed, but loss is the primary metric.

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a validation step using MCTS data.
        """
        total_loss, policy_loss, value_loss, policy_logits, value_pred = self._calculate_loss(batch)
        policy_targets = batch[1].to(self.device).float() # Get targets for potential accuracy calc

        # Logging
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch[0].size(0))
        self.log('val_policy_loss', policy_loss, on_step=False, on_epoch=True, logger=True, batch_size=batch[0].size(0))
        self.log('val_value_loss', value_loss, on_step=False, on_epoch=True, logger=True, batch_size=batch[0].size(0))

        # --- Optional: Calculate Top-1 Accuracy ---
        # Check if the predicted move with the highest probability
        # matches the move with the highest probability in the target distribution.
        predicted_move_idx = torch.argmax(policy_logits, dim=1)
        target_move_idx = torch.argmax(policy_targets, dim=1)
        accuracy = (predicted_move_idx == target_move_idx).float().mean()
        self.log('val_top1_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch[0].size(0))

        return total_loss

    # configure_optimizers can be inherited from BaseChessNN or overridden if specific settings are needed
    # def configure_optimizers(self):
    #     # ... custom optimizer/scheduler for MCTS training ...
    #     return super().configure_optimizers() # Or return custom dict