#!/usr/bin/env python3
import sys
import os
from datetime import datetime
import random

# Ensure parent directory is in sys.path for project module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hdf5plugin 
import h5py
import numpy as np
import matplotlib.pyplot as plt
import chess
from core.encoding import ChessEncoder

def display_chess_board(board_obj: chess.Board):
    """Prints a chess.Board object in a readable format."""
    print(board_obj)
    print("")

def visualize_encoded_planes(encoded_position_planes: np.ndarray, position_idx: int):
    """Visualizes the 18 planes of an encoded chess position using matplotlib."""
    if encoded_position_planes.shape != (18, 8, 8):
        print(f"Error: Expected shape (18, 8, 8) for encoded planes, got {encoded_position_planes.shape}")
        return

    plt.figure(figsize=(18, 6))
    plt.suptitle(f"Encoded Planes for Position Index {position_idx}", fontsize=16)
    
    plane_titles = [
        'White Pawns', 'White Knights', 'White Bishops', 'White Rooks', 'White Queens', 'White Kings',
        'Black Pawns', 'Black Knights', 'Black Bishops', 'Black Rooks', 'Black Queens', 'Black Kings',
        'Repetitions (Plane 1)', 'Repetitions (Plane 2)',
        'Side to Move (White=1)', 
        'White Can Castle Kingside', 'White Can Castle Queenside',
        'Black Can Castle Kingside', 'Black Can Castle Queenside'
    ]
    if len(plane_titles) > encoded_position_planes.shape[0]:
        plane_titles = plane_titles[:encoded_position_planes.shape[0]]

    for i in range(encoded_position_planes.shape[0]):
        plt.subplot(2, (encoded_position_planes.shape[0] + 1) // 2, i + 1)
        plt.imshow(encoded_position_planes[i], cmap='viridis')
        plt.title(plane_titles[i] if i < len(plane_titles) else f"Plane {i+1}", fontsize=10)
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def process_and_display_hdf5(hdf5_file: h5py.File, chess_encoder: ChessEncoder):
    """Processes an open HDF5 file, displaying metadata and samples of its content."""
    print("\n--- File Metadata ---")
    games_count = hdf5_file.attrs.get('games_count', 'N/A')
    creation_timestamp = hdf5_file.attrs.get('creation_timestamp', None)
    source_description = hdf5_file.attrs.get('source', 'N/A')
    
    creation_time_str = 'N/A'
    if isinstance(creation_timestamp, (int, float)):
        try:
            creation_time_str = datetime.fromtimestamp(creation_timestamp).strftime('%Y-%m-%d %H:%M:%S UTC')
        except TypeError:
             creation_time_str = str(creation_timestamp)

    print(f"  Games Count: {games_count}")
    print(f"  Creation Time: {creation_time_str}")
    print(f"  Source: {source_description}")
    
    required_datasets = ['positions', 'moves', 'values']
    for ds_name in required_datasets:
        if ds_name not in hdf5_file:
            print(f"\nError: Required dataset '{ds_name}' not found in HDF5 file.")
            return

    positions_ds = hdf5_file['positions']
    moves_ds = hdf5_file['moves']
    values_ds = hdf5_file['values']
    
    total_positions = len(positions_ds)
    print(f"\n--- Dataset Information ---")
    print(f"  Positions: {total_positions:,} (Shape: {positions_ds.shape})")
    print(f"  Moves:     {len(moves_ds):,} (Shape: {moves_ds.shape})")
    print(f"  Values:    {len(values_ds):,} (Shape: {values_ds.shape})")

    if not (total_positions == len(moves_ds) == len(values_ds)):
        print("\nWARNING: Dataset lengths mismatch! This could indicate corrupted data.")
    else:
        print("  Dataset lengths are consistent.")
    
    if total_positions == 0:
        print("\nNo positions found in the dataset. Cannot display samples.")
        return

    num_samples_to_show = min(5, total_positions)
    indices_to_show = random.sample(range(total_positions), num_samples_to_show)
    indices_to_show.sort()
    
    print(f"\n--- Displaying {num_samples_to_show} Random Samples ---")
    
    for i, data_idx in enumerate(indices_to_show):
        print(f"\nSample {i+1}/{num_samples_to_show} (Overall Index: {data_idx})")
        
        encoded_planes = positions_ds[data_idx]
        print(f"  Encoded Position Shape: {encoded_planes.shape}")

        try:
            decoded_board = chess_encoder.decode_board(encoded_planes)
            print("  Decoded Board:")
            display_chess_board(decoded_board)
            
            if decoded_board.is_valid():
                print(f"    FEN: {decoded_board.fen()}")
            else:
                print("    Warning: Decoded board is not in a valid state according to chess rules.")
        except Exception as e:
            print(f"  Error decoding board: {e}")
            if input("Board decoding failed. Visualize raw encoded planes? (y/n): ").lower().startswith('y'):
                visualize_encoded_planes(encoded_planes, data_idx)
        
        encoded_move_val = moves_ds[data_idx]
        try:
            decoded_chess_move = chess_encoder.decode_move(encoded_move_val) 
            print(f"  Decoded Move: {decoded_chess_move.uci() if isinstance(decoded_chess_move, chess.Move) else str(decoded_chess_move)}")
        except Exception as e:
            print(f"  Error decoding move: {e}")
            print(f"  Raw Encoded Move Value: {encoded_move_val}")
        
        position_value = values_ds[data_idx]
        print(f"  Position Value: {position_value[0] if isinstance(position_value, (list, np.ndarray)) and len(position_value)>0 else position_value:.4f}")
        print("-" * 50)
        
        if i < len(indices_to_show) - 1:
            if input("Press Enter for next sample, or 'q' to quit: ").strip().lower() == 'q':
                break

def main():
    if len(sys.argv) < 2:
        print("Usage: python testHDF5.py <path_to_hdf5_file>")
        sys.exit(1)
        
    file_path_arg = sys.argv[1]
    
    # Basic path correction for WSL if path starts with 'mnt/' but not '/'
    if file_path_arg.startswith('mnt/') and not file_path_arg.startswith('/'):
        file_path_arg = '/' + file_path_arg
        print(f"Path interpreted as WSL path: {file_path_arg}")
    
    if not os.path.exists(file_path_arg):
        print(f"Error: HDF5 file not found at '{file_path_arg}'")
        sys.exit(1)
        
    print(f"Attempting to open HDF5 file: {file_path_arg}")
    
    chess_encoder_instance = ChessEncoder()
    
    try:
        with h5py.File(file_path_arg, 'r') as hdf_file_obj:
            process_and_display_hdf5(hdf_file_obj, chess_encoder_instance)
    except OSError as e:
        # This OSError handling for plugin issues might be too specific or outdated
        # depending on h5py and plugin versions. Modern h5py often handles this better.
        print(f"OSError opening file: {e}")
        if "plugin" in str(e).lower(): # More generic check for plugin related error messages
            print("\nPotential HDF5 plugin issue detected. This might be due to missing filter plugins (e.g., Blosc, LZF).")
            print("Ensure HDF5 filter plugins are correctly installed and accessible in your Python environment.")
            print("You can try installing 'hdf5plugin' if you haven't: pip install hdf5plugin")
        # The attempts to re-open with 'core' driver or HDF5_DISABLE_VERSION_CHECK are often workarounds
        # for specific, older issues and might not be universally helpful or could hide problems.
        # It's generally better to ensure the environment and plugins are set up correctly.
    except Exception as e_general:
        print(f"An unexpected error occurred: {e_general}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
