import os
import torch
import chess.pgn
import datetime
import re
from typing import Optional, Dict

# Input validation utilities
def get_int_input(prompt, default=None, min_val=None):
    """Get integer input with validation and optional default."""
    while True:
        try:
            user_input = input(prompt)
            if not user_input and default is not None:
                return default
            value = int(user_input)
            if min_val is not None and value < min_val:
                print(f"Value must be >= {min_val}")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_float_input(prompt, default=None):
    """Get float input with validation and optional default."""
    while True:
        try:
            user_input = input(prompt)
            if not user_input and default is not None:
                return default
            return float(user_input)
        except ValueError:
            print("Invalid input. Please enter a number.")

# File selection utilities
def select_file(directory, extension, prompt_name):
    """Select a file from directory with given extension."""
    if not os.path.isdir(directory):
        print(f"Directory '{directory}' not found.")
        return None
    
    available_files = sorted([f for f in os.listdir(directory) if f.endswith(extension)])
    if not available_files:
        print(f"No {extension} files found in {directory}")
        return None
    
    print(f"\nAvailable {prompt_name} files:")
    for i, fname in enumerate(available_files):
        print(f"{i+1}. {fname}")
    
    choice = get_int_input(f"Select {prompt_name} file (enter number): ", min_val=1)
    if 1 <= choice <= len(available_files):
        return os.path.join(directory, available_files[choice - 1])
    else:
        print("Invalid selection.")
        return None

# Model utilities
def select_and_load_model(device):
    """Select and load a model, returning the state dict."""
    model_path = select_file("models", ".pth", "model")
    if not model_path:
        return None, None
    
    try:
        model_name = os.path.basename(model_path)
        print(f"Loading model: {model_name}...")
        model_state_dict = torch.load(model_path, map_location=device)
        print("Model loaded successfully.")
        return model_state_dict, model_name
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# Training configuration utilities
def get_training_params():
    """Get training parameters with defaults."""
    return {
        'epochs': get_int_input("Enter number of epochs (default 10): ", 10, 1),
        'chunks_per_batch': get_int_input("Enter chunks per batch (default 4): ", 4, 1),
        'learning_rate': get_float_input("Enter learning rate (default 0.0005): ", 0.0005),
        'val_split': get_float_input("Enter validation split (default 0.1): ", 0.1),
        'patience': get_int_input("Enter early stopping patience (default 5): ", 5, 1)
    }

# PGN Handling Utilities
def create_pgn_game_object(white_name: str, black_name: str, event: str, site: str = "Local Machine", additional_headers: Optional[Dict[str, str]] = None) -> chess.pgn.Game:
    """Creates a PGN game object with standard headers."""
    game_pgn = chess.pgn.Game()
    game_pgn.headers["Event"] = event
    game_pgn.headers["Site"] = site
    game_pgn.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    game_pgn.headers["Round"] = "1" # Default or make it a parameter
    game_pgn.headers["White"] = white_name
    game_pgn.headers["Black"] = black_name
    if additional_headers:
        for key, value in additional_headers.items():
            game_pgn.headers[key] = str(value) # Ensure values are strings for PGN
    return game_pgn

def save_pgn_file(game_pgn: chess.pgn.Game, base_save_dir: str, filename_prefix: str, p1_name: str, p2_name: str, include_timestamp: bool = True) -> Optional[str]:
    """Saves a PGN game to a file with a standardized name, returns the filepath."""
    try:
        os.makedirs(base_save_dir, exist_ok=True)
        
        safe_p1 = re.sub(r'[\\/*?:"<>|]', "", str(p1_name).replace('.pth',''))
        safe_p2 = re.sub(r'[\\/*?:"<>|]', "", str(p2_name).replace('.pth',''))
        
        timestamp_str = f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}" if include_timestamp else ""
        
        pgn_filename = f"{filename_prefix}_{safe_p1}_vs_{safe_p2}{timestamp_str}.pgn"
        pgn_filepath = os.path.join(base_save_dir, pgn_filename)
        
        print(f"\nSaving PGN game to {pgn_filepath}...")
        with open(pgn_filepath, "w", encoding="utf-8") as f:
            exporter = chess.pgn.FileExporter(f)
            game_pgn.accept(exporter)
        print("PGN game saved successfully.")
        return pgn_filepath
    except Exception as e:
        print(f"Error saving PGN file: {e}")
        return None 