import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import torch
import torch.nn as nn
import tqdm
import time
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
from neural_network.neuralNetworkSP import MCTSChessNN
import traceback

from self_play.datasetSP import ChessSPDataset
from self_play.selfPlay import SelfPlayChess
from core.encoding import ChessEncoder
import shutil
import gc


def train_mcts_model(
    model: MCTSChessNN, # Expecting the MCTS network
    mcts_dataset: Dataset, # Expecting a PyTorch Dataset (like ChessSPDataset)
    output_dir: str = ".mcts_checkpoints",
    model_filename_prefix: str = "mcts_model",
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-3, # Can be managed by Lightning/model's configure_optimizers
    val_split: float = 0.1,
    num_workers: int = None,
    resume_from_checkpoint: str = None
    ):
    """ Trains the MCTSChessNN model using PyTorch Lightning. """
    print(f"--- Starting MCTS Model Training (PyTorch Lightning) ---")
    os.makedirs(output_dir, exist_ok=True)

    if not (0 < val_split < 1): val_split = 0.1
    dataset_size = len(mcts_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError(f"Dataset too small ({dataset_size}) for validation split.")

    train_dataset, val_dataset = random_split(mcts_dataset, [train_size, val_size])
    print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    if num_workers is None: num_workers = min(8, os.cpu_count() or 1)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        filename=f'{model_filename_prefix}-{{epoch:02d}}-{{val_loss:.4f}}',
        save_top_k=1, monitor='val_loss', mode='min'
    )
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, mode='min', min_delta=0.0001
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision='16-mixed' if torch.cuda.is_available() else 32,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        logger=pl.loggers.TensorBoardLogger("lightning_logs/", name=model_filename_prefix),
        log_every_n_steps=max(10, len(train_loader)//20),
        benchmark=True,
    )

    print("Starting trainer.fit()...")
    start_run_time = time.time()
    try:
        trainer.fit(
            model=model, train_dataloaders=train_loader,
            val_dataloaders=val_loader, ckpt_path=resume_from_checkpoint
        )
        print(f"trainer.fit() finished. Duration: {time.time() - start_run_time:.2f}s")
    except Exception as e:
        print(f"Training error: {e}"); traceback.print_exc()
        return model, checkpoint_callback.best_model_path # Return current model state on error

    print(f"Training finished. Best model path: {checkpoint_callback.best_model_path}")
    # Add time tracking loading/logging if using MCTSChessNNWithTime
    if hasattr(model, 'total_training_time_sec'):
         print(f"Cumulative training time: {model.total_training_time_sec.item():.2f}s")

    return model, checkpoint_callback.best_model_path # Return trained model and best ckpt path

def run_mcts_training_loop(
    # --- Configuration Parameters ---
    num_iterations: int,
    games_per_iteration: int,
    epochs_per_iteration: int,
    mcts_simulations: int,
    batch_size: int,
    learning_rate: float, # Initial LR, might be adjusted by scheduler
    checkpoint_dir: str,
    iteration_prefix: str,
    start_iteration: int,
    # --- Objects ---
    initial_model: MCTSChessNN, # Pass the instantiated model object
    encoder: ChessEncoder,
    ):
    """
    Runs the main MCTS self-play training loop (Generate -> Train -> Repeat).
    """
    model = initial_model
    current_model_path = None
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Load state if resuming ---
    if start_iteration > 0:
        resume_path = os.path.join(checkpoint_dir, f"{iteration_prefix}-{start_iteration-1}.ckpt")
        if os.path.exists(resume_path):
            print(f"Attempting to load loop state from: {resume_path}")
            try:
                checkpoint = torch.load(resume_path, map_location='cpu')
                model.load_state_dict(checkpoint['state_dict'])
                # Optional: Load time if using MCTSChessNNWithTime
                # if hasattr(model, 'on_load_checkpoint'): model.on_load_checkpoint(checkpoint)
                current_model_path = resume_path
                print(f"Resumed weights for Iteration {start_iteration}.")
            except Exception as e:
                print(f"Warning: Failed to load checkpoint state: {e}. Starting fresh.")
                start_iteration = 0 # Reset start if load failed
        else:
            print(f"Warning: Resume checkpoint not found: {resume_path}. Starting fresh.")
            start_iteration = 0 # Reset start if not found

    # --- Main Loop ---
    for iteration in range(start_iteration, num_iterations):
        print(f"\n===== Self-Play Iteration {iteration + 1} / {num_iterations} =====")

        # 1. Generate Data
        print(f"--- Generating {games_per_iteration} games (MCTS Sims: {mcts_simulations}) ---")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        model.eval()

        self_player = SelfPlayChess(model, encoder, games_per_iteration, mcts_simulations=mcts_simulations, num_workers=8)
        generated_data_container = self_player.run_self_play()

        if not hasattr(generated_data_container, 'positions') or len(generated_data_container.positions) == 0:
            print("Error: No self-play data generated. Skipping iteration.")
            continue

        try:
            mcts_dataset = ChessSPDataset(
                generated_data_container.positions,
                generated_data_container.policy_targets,
                generated_data_container.values
            )
        except Exception as e:
            print(f"Error creating dataset: {e}")
            continue

        # 2. Train Model
        print(f"\n--- Training on {len(mcts_dataset)} positions (Epochs: {epochs_per_iteration}) ---")
        # Ensure train_mcts_model is defined above or imported
        model, best_ckpt_path = train_mcts_model(
            model=model,
            mcts_dataset=mcts_dataset,
            output_dir=checkpoint_dir,
            model_filename_prefix=f"{iteration_prefix}-{iteration}",
            epochs=epochs_per_iteration,
            batch_size=batch_size,
            learning_rate=learning_rate, # Pass LR, though scheduler might override
            # resume_from_checkpoint=None # Start fresh trainer session each iteration
        )

        # Save/Update iteration checkpoint path
        if best_ckpt_path and os.path.exists(best_ckpt_path):
            iter_save_path = os.path.join(checkpoint_dir, f"{iteration_prefix}-{iteration}.ckpt")
            try:
                shutil.copy(best_ckpt_path, iter_save_path)
                current_model_path = iter_save_path
                print(f"Saved iteration checkpoint: {iter_save_path}")
            except Exception as e:
                print(f"Warning: Could not copy best ckpt {best_ckpt_path}: {e}")
                current_model_path = best_ckpt_path # Fallback
        else:
            print("Warning: No best checkpoint path from training. Saving current state.")
            fallback_save_path = os.path.join(checkpoint_dir, f"{iteration_prefix}-{iteration}-fallback.ckpt")
            # Need a trainer instance to save - simplest is often just model state_dict
            # torch.save(model.state_dict(), fallback_save_path) # Save state_dict as fallback
            # Or use minimal trainer to save full PL checkpoint
            temp_trainer = pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False)
            temp_trainer.save_checkpoint(fallback_save_path, weights_only=False)
            current_model_path = fallback_save_path
            print(f"Saved fallback state: {fallback_save_path}")


        # Clean up memory
        del generated_data_container
        del mcts_dataset
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print("\n--- Self-Play Training Loop Function Finished ---")
    return model # Return the final trained model object

def train_sp(model, dataset, epochs=10, batch_size=64, learning_rate=1e-3):
    """Optimized training using GPU"""
    # Move model to GPU for training if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.train()
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=8,  # Use multiple workers for data loading
        pin_memory=(device=='cuda')
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5
    )
    
    for epoch in range(epochs):
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_batches = 0
        
        for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            positions, move_indices, values = batch
            
            # Move data to the same device as the model
            positions = positions.to(device).float()
            move_indices = move_indices.to(device).long()
            values = values.to(device).float()
            
            optimizer.zero_grad()
            
            # Forward pass
            policy_out, value_out = model(positions)
            
            # Calculate losses
            policy_loss = criterion_policy(policy_out, move_indices)
            value_loss = criterion_value(value_out, values)
            loss = policy_loss + 0.5 * value_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_batches += 1
        
        # Calculate average losses
        avg_loss = total_loss / total_batches
        avg_policy_loss = total_policy_loss / total_batches
        avg_value_loss = total_value_loss / total_batches
        
        # Update learning rate based on loss
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Loss: {avg_loss:.4f} (Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f})")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    return model