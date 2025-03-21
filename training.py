import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import os

from dataset import ChessDataset
from neuralNetwork import ChessNN

# TODO - transformer engine and FP8 usage?

def train_model(positions, moves, values, epochs=100, batch_size=2048, 
                learning_rate=1e-3, val_split=0.1):
    """
    Train a chess model with pre-encoded positions
    
    Args:
        positions: numpy array of encoded positions
        moves: numpy array of encoded moves
        values: numpy array of position evaluations
        epochs: number of training epochs
        batch_size: batch size for training
        learning_rate: learning rate for optimizer
        val_split: fraction of data to use for validation
    
    Returns:
        The trained ChessNN model
    """
    # Determine input channels from the positions
    if len(positions.shape) > 3:  # Batch of positions
        input_channels = positions.shape[1]
    else:  # Single position
        input_channels = positions.shape[0]
    
    # Ensure values have correct shape (N, 1)
    if isinstance(values, np.ndarray):
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        elif values.ndim > 2:
            values = values.reshape(-1, 1)
    
    # Output debug info about position count, input channels, position and value shapes
    print(f"Training with {len(positions)} positions")
    print(f"Input channels: {input_channels}")
    print(f"Positions shape: {positions.shape}")
    print(f"Values shape: {values.shape}")
    
    # Create dataset
    dataset = ChessDataset(positions, moves, values)
    
    # Split into train and validation
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=min(8, os.cpu_count() or 1),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size*2,
        shuffle=False,
        num_workers=min(8, os.cpu_count() or 1),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Create model with appropriate input channels
    model = ChessNN(input_channels=input_channels, learning_rate=learning_rate)
    
    # Setup temporary directory for model checkpoints
    output_dir = ".temp_model"
    os.makedirs(output_dir, exist_ok=True)
    
    # Training with pl.Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=16, # FP16 training
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=output_dir,
                filename='best-model',
                save_top_k=1,
                monitor='val_loss',
                mode='min'
            ),
            pl.callbacks.LearningRateMonitor()
        ]
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Bset model is the last checkpoint
    best_model = model
    
    # Clean up temporary directory
    import shutil
    shutil.rmtree(output_dir, ignore_errors=True)
    
    # Return best found model during training to the user
    return best_model