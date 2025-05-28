import h5py
import hdf5plugin
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info, Subset, Sampler
import os
import numpy as np
import time
import traceback
import math
import random

def hdf5_worker_init(worker_id):
    """Opens the HDF5 file once per worker process."""
    worker_info = get_worker_info()
    dataset_obj = worker_info.dataset

    # Handle Subset objects created by random_split
    original_dataset = dataset_obj.dataset if isinstance(dataset_obj, Subset) else dataset_obj

    # Reset handle attribute for this worker first
    original_dataset._worker_hdf5_handle = None

    if not hasattr(original_dataset, 'hdf5_path') or not original_dataset.hdf5_path: # Check for valid hdf5_path
        print(f"[Worker {worker_id}] ERROR: Invalid passed hdf5_path attribute.")
        return

    try:
        # Store handle on original dataset object
        original_dataset._worker_hdf5_handle = h5py.File(original_dataset.hdf5_path, 'r')
    except Exception as e:
        print(f"[Worker {worker_id}] Error opening HDF5 file {original_dataset.hdf5_path}: {e}")
        traceback.print_exc()
        original_dataset._worker_hdf5_handle = None

class ChunkedHDF5Dataset(Dataset):
    """
    Dataset that treats each HDF5 chunk as a single item.
    Assumes all datasets ('positions', 'moves', 'values') have
    the same number of samples and compatible chunking along the first axis.
    """
    def __init__(self, data_source):
        if not isinstance(data_source, str) or not data_source.endswith('.hdf5'):
            raise TypeError("data_source must be a string path to an HDF5 file.")
        if not os.path.exists(data_source):
            raise FileNotFoundError(f"HDF5 file not found: {data_source}")

        self.hdf5_path = data_source
        self._worker_hdf5_handle = None  # Set by hdf5_worker_init
        self._total_samples = 0
        self.samples_per_chunk = 0
        self._num_chunks = 0
        self.position_chunk_shape = None # Store for reference

        print(f"Initializing Chunked HDF5 Dataset: {self.hdf5_path}")
        try:
            # Open file and read metadata
            with h5py.File(self.hdf5_path, 'r') as f:
                required_keys = ['positions', 'moves', 'values']
                if not all(key in f for key in required_keys):
                    missing = [key for key in required_keys if key not in f]
                    raise KeyError(f"HDF5 file missing required dataset(s): {missing}")

                # Get total number of samples (use 'moves' as reference)
                self._total_samples = f['moves'].shape[0]
                if self._total_samples == 0:
                    raise ValueError("No samples found in file.")

                # Determine samples per chunk
                # Assuming chunking is defined and consistent for the first dimension
                pos_chunks = f['positions'].chunks
                if pos_chunks is None:
                     raise ValueError("Positions dataset is not chunked.")
                self.samples_per_chunk = pos_chunks[0]
                self.position_chunk_shape = pos_chunks # (Samples, C, H, W)

                # Verify if other chunks match
                mov_chunks = f['moves'].chunks
                val_chunks = f['values'].chunks
                if mov_chunks is None or mov_chunks[0] != self.samples_per_chunk:
                     print(f"Warning: Moves chunk size ({mov_chunks}) doesn't match positions ({self.samples_per_chunk}). Using positions chunk size.")
                if val_chunks is None or val_chunks[0] != self.samples_per_chunk:
                     print(f"Warning: Values chunk size ({val_chunks}) doesn't match positions ({self.samples_per_chunk}). Using positions chunk size.")


                if self.samples_per_chunk <= 0:
                     raise ValueError("Invalid samples_per_chunk from file.")

                # Calculate the number of chunks - dataset length
                self._num_chunks = math.ceil(self._total_samples / self.samples_per_chunk)

                print(f"  Total samples: {self._total_samples}")
                print(f"  Detected samples per chunk (from 'positions'): {self.samples_per_chunk}")
                print(f"  Position chunk shape: {self.position_chunk_shape}")
                print(f"  Total number of chunks (Dataset length): {self._num_chunks}")

        except Exception as e:
            print(f"Error reading metadata from HDF5 file {self.hdf5_path}: {e}")
            traceback.print_exc()
            raise # Re-raise the exception

        print("Chunked HDF Dataset initialization complete.")


    def __len__(self):
        """Returns the number of CHUNKS in the dataset."""
        return self._num_chunks

    def __getitem__(self, chunk_idx):
        """
        Returns the data in the chunk specified by chunk_idx.

        Args:
            chunk_idx (int): The index of the chunk to retrieve (0 to len(self)-1).

        Returns:
            tuple: (pos_chunk_np, mov_chunk_np, val_chunk_np)
                   NumPy arrays containing data for the whole chunk.
        """
        if not 0 <= chunk_idx < self._num_chunks: # If chunk_idx is out of range
            raise IndexError(f"Chunk index {chunk_idx} out of range for {self._num_chunks} chunks.")

        # Ensure worker handle is available
        hdf5_file = getattr(self, '_worker_hdf5_handle', None)
        temp_handle = None
        worker_id_str = f"Worker {get_worker_info().id}" if get_worker_info() else "Main Process"

        if hdf5_file is None:
            # Fallback: Open temporarily (slower backup version)
            try:
                temp_handle = h5py.File(self.hdf5_path, 'r')
                hdf5_file = temp_handle
            except Exception as e:
                 print(f"[{worker_id_str}] Failed to open HDF5 {self.hdf5_path} in __getitem__ for chunk_idx {chunk_idx}. Error: {e}")
                 raise RuntimeError(f"Failed to open HDF5 file for chunk index {chunk_idx}") from e

        if hdf5_file is None: # If no file
             raise RuntimeError(f"[{worker_id_str}] Failed to get file handle for chunk index {chunk_idx}")

        # Calculate sample start and end indices for the chunk
        start_sample_idx = chunk_idx * self.samples_per_chunk
        # Ensure end index doesn't exceed total samples (for potential partial chunk)
        end_sample_idx = min(start_sample_idx + self.samples_per_chunk, self._total_samples)

        try:
            # Read chunk slices
            pos_chunk = hdf5_file['positions'][start_sample_idx:end_sample_idx]
            mov_chunk = hdf5_file['moves'][start_sample_idx:end_sample_idx]
            val_chunk = hdf5_file['values'][start_sample_idx:end_sample_idx]

            # Ensure value chunk has shape (N, 1) even if read as (N,)
            if val_chunk.ndim == 1:
                 val_chunk = val_chunk[:, np.newaxis]

            return pos_chunk, mov_chunk, val_chunk

        except Exception as e:
            print(f"[{worker_id_str}] ERROR reading chunk {chunk_idx} (samples {start_sample_idx}:{end_sample_idx}) from HDF5 {self.hdf5_path}: {e}")
            traceback.print_exc()
            raise IndexError(f"Failed to read chunk index {chunk_idx} from HDF5.") from e
        finally:
            if temp_handle is not None: # If file was opened - close handle
                try:
                    temp_handle.close()
                except Exception:
                    pass

def chunked_collate(batch_list):
    """
    Collates a list of chunks into a single batch tensor.
    'batch_list' is a list where each element is the tuple
    (pos_chunk_np, mov_chunk_np, val_chunk_np) returned by __getitem__.

    Args:
        batch_list (list): A list of tuples, e.g.,
                           [ (pos_chunk_0, mov_chunk_0, val_chunk_0),
                             (pos_chunk_1, mov_chunk_1, val_chunk_1), ... ]

    Returns:
        tuple: (pos_batch_tensor, mov_batch_tensor, val_batch_tensor)
               PyTorch tensors ready for the model.
    """
    try:
        # Separate the components across the list of chunks
        pos_chunks, mov_chunks, val_chunks = zip(*batch_list)

        # Concatenate the chunks along the sample dimension
        # Example: if batch_list has 2 chunks of 2048 samples,
        # resulting pos_batch_np will have shape (4096, 18, 8, 8)
        pos_batch_np = np.concatenate(pos_chunks, axis=0)
        mov_batch_np = np.concatenate(mov_chunks, axis=0)
        val_batch_np = np.concatenate(val_chunks, axis=0) # Should already be (N, 1)

        # --- Convert to PyTorch Tensors ---
        # Perform type conversions needed for the model
        # Example: Convert uint8 positions to float and normalize
        pos_batch_tensor = torch.from_numpy(pos_batch_np).float()

        # Convert moves (int16) potentially to long for embedding layers or loss functions
        mov_batch_tensor = torch.from_numpy(mov_batch_np).long()

        # Convert values (float32)
        val_batch_tensor = torch.from_numpy(val_batch_np).float() # Already float32
        
        # Check val_batch_tensor values
        if val_batch_tensor.ndim == 1:
            val_batch_tensor = val_batch_tensor[:, np.newaxis] # Ensure shape is (N, 1)
            print(f"  Value batch reshaped to: {val_batch_tensor.shape}")
            
        # # Output samples
        # print(f"  Sample values (first 5):")
        # print(f"    Positions: {pos_batch_tensor[:1]}")
        # print(f"    Moves:     {mov_batch_tensor[:1]}")
        
        # # Output full float values with 3 decimal places
        # print(f"  Full float values (first 5):")
        # print(f"    Values: {val_batch_tensor[:5].float().numpy()}")

        return pos_batch_tensor, mov_batch_tensor, val_batch_tensor

    except Exception as e:
        print(f"Error in chunked_collate_fn: {e}")
        traceback.print_exc()
        # Print details about the batch items that caused the error
        print("Problematic batch list items (before concatenation):")
        for i, item in enumerate(batch_list):
            print(f"  Item {i} (Chunk):")
            try:
                p, m, v = item
                print(f"    Pos type: {type(p)}, shape: {getattr(p, 'shape', 'N/A')}, dtype: {getattr(p, 'dtype', 'N/A')}")
                print(f"    Move type: {type(m)}, shape: {getattr(m, 'shape', 'N/A')}, dtype: {getattr(m, 'dtype', 'N/A')}")
                print(f"    Value type: {type(v)}, shape: {getattr(v, 'shape', 'N/A')}, dtype: {getattr(v, 'dtype', 'N/A')}")
            except Exception as item_e:
                print(f"    Error inspecting item {i}: {item_e}")
        return None # Or raise e

class ChunkSampler(Sampler):
    """ Samples chunk indices. Used for train/val splitting. """
    def __init__(self, num_chunks, indices):
        self.num_chunks = num_chunks
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)