import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import os
import random
import shutil
import traceback
# Importing hdfDataset and methods
from existing_games.hdfDataset import ChunkedHDF5Dataset, hdf5_worker_init, chunked_collate, ChunkSampler
from neural_network.neuralNetwork import ChessNN

def train_model(data_path, epochs=100,
                        chunks_per_batch=2,
                        learning_rate=0.3e-3, val_split=0.1,
                        initial_model_path=None, patience=5):
    """
    Train a chess model using chunked HDF5 loading.
    
    Args:
        data_path: Path to the HDF5 data file.
        epochs: Number of epochs to train.
        chunks_per_batch: Number of HDF5 chunks per batch.
        learning_rate: Optimizer learning rate.
        val_split: Fraction of data to use for validation.
        initial_model_path: Optional path to a pre-trained model to continue training.
        patience: Early stopping patience (epochs with no val_loss improvement).
    """
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = True # For Ampere and newer GPUs
    
    try:
        dataset = ChunkedHDF5Dataset(data_path)
    except Exception as e: # Generic exception for dataset loading
        print(f"Error initializing chunked dataset from {data_path}: {e}")
        return None

    if dataset.position_chunk_shape is None or len(dataset.position_chunk_shape) < 4:
         print(f"Error: Position chunk shape {dataset.position_chunk_shape} not found or invalid in {data_path}.")
         return None
    input_channels = dataset.position_chunk_shape[1] # (N, C, H, W)
    num_chunks_total = len(dataset)
    # samples_per_chunk = dataset.samples_per_chunk

    print(f"Training with Chunked Dataset: {data_path}")
    print(f"  Total Chunks: {num_chunks_total}, Samples/Chunk: {dataset.samples_per_chunk}, Input Channels: {input_channels}")
    print(f"  DataLoader 'batch_size' corresponds to chunks per GPU batch.")

    if not (0 < val_split < 1):
        print(f"Warning: Invalid val_split {val_split}. Using default 0.1.")
        val_split = 0.1

    chunk_indices = list(range(num_chunks_total))
    random.shuffle(chunk_indices)

    split_idx = int(num_chunks_total * (1 - val_split))
    train_chunk_indices = chunk_indices[:split_idx]
    val_chunk_indices = chunk_indices[split_idx:]
    print(f"  Train Chunks: {len(train_chunk_indices)}, Validation Chunks: {len(val_chunk_indices)}")
    
    train_sampler = ChunkSampler(num_chunks_total, train_chunk_indices)
    val_sampler = ChunkSampler(num_chunks_total, val_chunk_indices)

    num_workers = min(12, os.cpu_count() or 1)
    prefetch_factor = 2
    print(f"Using {num_workers} workers and prefetch_factor: {prefetch_factor} for DataLoaders.")

    train_loader = DataLoader(
        dataset,
        batch_size=chunks_per_batch, # Number of chunks
        sampler=train_sampler,
        shuffle=False, # Sampler handles shuffle of chunks
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=hdf5_worker_init,
        collate_fn=chunked_collate,
        drop_last=True # Consistent batch sizes
    )

    val_loader = DataLoader(
        dataset,
        batch_size=chunks_per_batch * 2, # Larger batch for validation if enough VRAM
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=hdf5_worker_init,
        collate_fn=chunked_collate
    )

    model = None
    if initial_model_path and os.path.exists(initial_model_path):
        print(f"Loading initial model from: {initial_model_path}")
        try:
            model = ChessNN.load_from_checkpoint(
                initial_model_path,
                input_channels=input_channels,
                learning_rate=learning_rate
            )
            print("Successfully loaded model from Lightning checkpoint.")
        except Exception:
            print(f"Failed to load as Lightning checkpoint, attempting to load as raw state_dict.")
            try:
                model_instance = ChessNN(input_channels=input_channels, learning_rate=learning_rate)
                model_instance.load_state_dict(torch.load(initial_model_path, map_location=lambda storage, loc: storage))
                model = model_instance
                print("Successfully loaded model from raw state_dict.")
            except Exception as e_state_dict:
                print(f"Error loading initial model state_dict: {e_state_dict}")
    
    if model is None:
        print("Creating new model instance.")
        model = ChessNN(input_channels=input_channels, learning_rate=learning_rate)

    output_dir = ".temp_model_checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        filename='best-model-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min',
        min_delta=0.0005 # Minimum change to keep as better model
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision='16-mixed' if torch.cuda.is_available() else "32-true", # 32-true for CPU
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        logger=pl.loggers.TensorBoardLogger("lightning_logs/", name="chess_model_chunked"),
        log_every_n_steps=25,
        benchmark=True, # Enables cudnn.benchmark
        gradient_clip_val=0.5, # Try to prevent exploding gradients
    )

    print("Starting training...")
    final_model_to_return = None
    try:
        trainer.fit(model, train_loader, val_loader)
        # If fit completes, load best model from checkpoint_callback
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path and os.path.exists(best_model_path):
            print(f"Training completed. Loading best model from: {best_model_path}")
            final_model_to_return = ChessNN.load_from_checkpoint(
                best_model_path,
                input_channels=input_channels, # Ensure channels match
                learning_rate=learning_rate   # during load_from_checkpoint
            )
            print("Best model loaded")
        else:
            print("Training completed, but no best checkpoint found. Returning last state.")
            final_model_to_return = model # Fallback to the model state at end of training

    except Exception as e:
        print(f"An error occurred during training: {e}")
        traceback.print_exc()
        print("Attempting to salvage best model found...")
        best_model_path_on_error = checkpoint_callback.best_model_path
        if best_model_path_on_error and os.path.exists(best_model_path_on_error):
            try:
                print(f"Loading best model from (on error): {best_model_path_on_error}")
                final_model_to_return = ChessNN.load_from_checkpoint(
                    best_model_path_on_error,
                    input_channels=input_channels,
                    learning_rate=learning_rate
                )
                print("Successfully salvaged best checkpoint.")
            except Exception as e_load:
                print(f"Could not load checkpoint: {e_load}. Returning None or last known state.")
                final_model_to_return = model # Fallback to model before fit if salvage fails
        else:
            print("No best model checkpoint found. Returning model state before fit call or None.")
            final_model_to_return = model # None if model wasn't initialized
    finally:
        # Clean up temporary checkpoint directory if training happened
        if os.path.exists(output_dir):
            print(f"Cleaning temp checkpoint dir: {output_dir}")
            shutil.rmtree(output_dir, ignore_errors=True)
        else:
            print("No temp checkpoint directory to clean up.")

    return final_model_to_return
