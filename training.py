import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import torch
import torch.nn as nn
import tqdm

# Importing hdfDataset and methods
from hdfDataset import ChunkedHDF5Dataset, hdf5_worker_init, chunked_collate, ChunkSampler
from neuralNetwork import ChessNN

def train_model(data_path, epochs=100,
                        chunks_per_batch=2,
                        learning_rate=0.3e-3, val_split=0.1):
    """
    Train a chess model using chunked HDF5 loading.
    """
    try:
        # Use the new chunked dataset
        dataset = ChunkedHDF5Dataset(data_path)
    except (FileNotFoundError, TypeError, IOError, KeyError, ValueError) as e:
        print(f"Error initializing chunked dataset: {e}")
        return None

    # Get input channels from dataset - shape derived from chunk shape stored in init
    if dataset.position_chunk_shape is None or len(dataset.position_chunk_shape) < 4:
         print(f"Error: Position chunk shape {dataset.position_chunk_shape} not found or invalid.")
         return None
    input_channels = dataset.position_chunk_shape[1] # (Samples, C, H, W)
    num_chunks_total = len(dataset)
    samples_per_chunk = dataset.samples_per_chunk

    print(f"Training with Chunked Dataset:")
    print(f"  Total Chunks: {num_chunks_total}")
    print(f"  Samples per Chunk: {samples_per_chunk}")
    print(f"  Input channels: {input_channels}")
    print(f"  DataLoader 'batch_size' means chunks per batch.")

    # --- Split into Train and Validation sets ---
    if not (0 < val_split < 1):
        print("Warning: Invalid val_split. Using 0.1.")
        val_split = 0.1

    chunk_indices = list(range(num_chunks_total))
    random.shuffle(chunk_indices) # Shuffle chunk order

    split_idx = int(num_chunks_total * (1 - val_split))
    train_chunk_indices = chunk_indices[:split_idx]
    val_chunk_indices = chunk_indices[split_idx:]

    num_train_chunks = len(train_chunk_indices)
    num_val_chunks = len(val_chunk_indices)
    print(f"  Train chunks: {num_train_chunks}, Validation chunks: {num_val_chunks}")
    
    # Custom ChunkSampler for correct chunk indices to each DataLoader
    train_sampler = ChunkSampler(num_chunks_total, train_chunk_indices)
    val_sampler = ChunkSampler(num_chunks_total, val_chunk_indices)


    # --- Create Data Loaders ---
    num_workers = min(10, os.cpu_count() or 1) # Worker adjustment
    print(f"Using {num_workers} workers for DataLoaders.")
    prefetch_factor = 2 # Prefetch factor adjustment
    print(f"Using prefetch_factor: {prefetch_factor}")

    # No shuffle - handled by sampler
    train_loader = DataLoader(
        dataset, # Full dataset instance
        batch_size=chunks_per_batch, # Number of chunks per batch (samples = chunks*samples_per_chunk)
        sampler=train_sampler,       # Sampler for chunk indices
        shuffle=False,               # Sampler for chunk ordering
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=hdf5_worker_init,
        collate_fn=chunked_collate, # Chunk collate function
        drop_last=True # Skipping last incomplete chunk
    )

    val_loader = DataLoader(
        dataset, # Full dataset instance
        batch_size=chunks_per_batch * 2, # Number of chunks per batch (samples = chunks*samples_per_chunk)
        sampler=val_sampler,         # Sampler for chunk indices
        shuffle=False,               # Sampler for chunk ordering
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=hdf5_worker_init,
        collate_fn=chunked_collate # Chunk collate functio
    )

    # --- Create Model ---
    # Effective batch size on GPU is chunks_per_batch * samples_per_chunk
    model = ChessNN(input_channels=input_channels, learning_rate=learning_rate)

    # --- Setup Training ---
    output_dir = ".temp_model"
    os.makedirs(output_dir, exist_ok=True)

    # Callbacks (monitoring validation accuracy))
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        filename='best-chunked-model-{epoch:02d}-{val_loss:.4f}', # Monitor loss
        save_top_k=1,
        monitor='val_loss', # Changed to val_loss
        mode='min'
    )
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss', # Changed to val_loss
        patience=20,
        mode='min',
        min_delta=0.0005
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision='16-mixed' if torch.cuda.is_available() else 32, # Mixed precision training
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        logger=pl.loggers.TensorBoardLogger("lightning_logs/", name="chess_model_chunked"),
        log_every_n_steps=10, # Log frequency
    )

    # --- Train Model ---
    print("Starting training...")
    try:
        trainer.fit(model, train_loader, val_loader)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        # Clean up on error
        import shutil
        
        # Ask user remove temp dir
        print(f"Remove temporary directory: {output_dir}? (y/n)")
        user_input = input().strip().lower()
        
        if user_input == 'y':
            shutil.rmtree(output_dir, ignore_errors=True)
            print(f"Temporary directory {output_dir} removed.")
        else:
            print(f"Temporary directory {output_dir} kept intact.")
        return None

    # --- Load Best Model found during training ---
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path and os.path.exists(best_model_path):
        print(f"Loading best chunked model from: {best_model_path}")
        best_model = ChessNN.load_from_checkpoint(
            best_model_path,
            input_channels=input_channels,
            learning_rate=learning_rate
        )
        print("Best model loaded successfully.")
    else:
        # Error handling
        best_model = model

    # Final cleanup
    import shutil
    shutil.rmtree(output_dir, ignore_errors=True)
    print("Cleaned up temporary chunked checkpoint directory.")

    return best_model


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
        num_workers=4,  # Use multiple workers for data loading
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