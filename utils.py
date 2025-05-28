import os
import torch

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