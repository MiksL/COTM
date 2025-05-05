import os
import shutil
import time
import torch
import glob
from datetime import datetime
from neural_network.neuralNetwork import ChessNN
from neural_network.neuralNetworkSP import MCTSChessNN

# Configuration
TEMP_MODEL_DIRS = [".temp_model", ".mcts_checkpoints"]  # Directories to search for checkpoints
MODELS_DIR = "models"  # Where to save the permanent models

class CheckpointManager:
    def __init__(self):
        # Create models directory if it doesn't exist
        os.makedirs(MODELS_DIR, exist_ok=True)
        
    def list_checkpoint_files(self, include_regular_pt=True):
        """Find checkpoint files in temp directories and sort by date modified."""
        checkpoint_files = []
        
        # Search in temporary directories
        for temp_dir in TEMP_MODEL_DIRS:
            if not os.path.exists(temp_dir):
                continue
                
            # Look for .ckpt (Lightning) files
            ckpt_files = glob.glob(os.path.join(temp_dir, "*.ckpt"))
            checkpoint_files.extend(ckpt_files)
            
            # Optionally include .pt/.pth files
            if include_regular_pt:
                pt_files = glob.glob(os.path.join(temp_dir, "*.pt")) + glob.glob(os.path.join(temp_dir, "*.pth"))
                checkpoint_files.extend(pt_files)
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return checkpoint_files
    
    def get_checkpoint_info(self, checkpoint_path):
        """Extract info about the checkpoint file."""
        # File info
        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        mod_time = datetime.fromtimestamp(os.path.getmtime(checkpoint_path))
        
        # Try to load some model info if possible
        model_info = "Unknown model format"
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Check if it's a PyTorch Lightning checkpoint
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                # Extract model type from state_dict keys
                keys = list(checkpoint["state_dict"].keys())
                if any("policy_head" in k for k in keys):
                    model_info = "Chess Neural Network"
                    
                    # Get validation metrics if available
                    if "callbacks" in checkpoint and "ModelCheckpoint" in checkpoint["callbacks"]:
                        checkpoint_callback = checkpoint["callbacks"]["ModelCheckpoint"]
                        if "best_model_score" in checkpoint_callback:
                            val_score = checkpoint_callback["best_model_score"].item()
                            model_info += f" (val_loss: {val_score:.4f})"
        except:
            model_info += " (failed to inspect)"
            
        info = {
            "file_path": checkpoint_path,
            "file_name": os.path.basename(checkpoint_path),
            "size_mb": size_mb,
            "modified": mod_time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_info": model_info
        }
        return info
    
    def save_checkpoint(self, checkpoint_path, new_name=None):
        """Save a checkpoint to the models directory with the given name."""
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint file {checkpoint_path} not found.")
            return False
            
        # Generate default name if none provided
        if not new_name:
            timestamp = time.strftime("%Y%m%d-%H%M")
            base_name = os.path.basename(checkpoint_path)
            if "best-model" in base_name:
                # Extract epoch and metrics if available
                parts = base_name.split("-")
                try:
                    epoch = next((p for p in parts if p.startswith("epoch")), "").replace("epoch", "")
                    val_loss = next((p for p in parts if p.startswith("val_loss")), "").replace("val_loss", "")
                    if epoch and val_loss:
                        new_name = f"model-e{epoch}-loss{val_loss}-{timestamp}.pth"
                    else:
                        new_name = f"model-{timestamp}.pth"
                except:
                    new_name = f"model-{timestamp}.pth"
            else:
                new_name = f"model-{timestamp}.pth"
            
        # Ensure .pth extension
        if not (new_name.endswith('.pth') or new_name.endswith('.pt')):
            new_name += '.pth'
            
        dest_path = os.path.join(MODELS_DIR, new_name)
        
        # Convert to state_dict if it's a Lightning checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                print(f"Converting Lightning checkpoint to state_dict for easier loading...")
                state_dict = checkpoint["state_dict"]
                torch.save(state_dict, dest_path)
                print(f"Saved state_dict to: {dest_path}")
                return True
            else:
                # Copy as-is (already a state dict or unknown format)
                shutil.copy2(checkpoint_path, dest_path)
                print(f"Copied checkpoint to: {dest_path}")
                return True
        except Exception as e:
            print(f"Error converting/saving checkpoint: {e}")
            # Fallback to direct copy
            try:
                shutil.copy2(checkpoint_path, dest_path)
                print(f"Copied checkpoint to: {dest_path}")
                return True
            except Exception as e:
                print(f"Error copying checkpoint: {e}")
                return False

    def clean_temp_directories(self):
        """Remove temp checkpoint directories."""
        for temp_dir in TEMP_MODEL_DIRS:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print(f"Removed temporary directory: {temp_dir}")
                except Exception as e:
                    print(f"Error removing {temp_dir}: {e}")

def main():
    manager = CheckpointManager()
    
    while True:
        print("\n===== Checkpoint Manager =====")
        print("1. List available checkpoints")
        print("2. Save a checkpoint to models directory")
        print("3. Clean up temporary directories")
        print("4. Quit")
        
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == "1":
            checkpoints = manager.list_checkpoint_files()
            if not checkpoints:
                print("\nNo checkpoint files found in temporary directories.")
                continue
                
            print(f"\nFound {len(checkpoints)} checkpoint files:")
            for i, checkpoint in enumerate(checkpoints):
                info = manager.get_checkpoint_info(checkpoint)
                print(f"{i+1}. {info['file_name']} ({info['size_mb']:.1f} MB)")
                print(f"   Modified: {info['modified']}")
                print(f"   Info: {info['model_info']}")
                
        elif choice == "2":
            checkpoints = manager.list_checkpoint_files()
            if not checkpoints:
                print("\nNo checkpoint files found to save.")
                continue
                
            print("\nAvailable checkpoints:")
            for i, checkpoint in enumerate(checkpoints):
                info = manager.get_checkpoint_info(checkpoint)
                print(f"{i+1}. {info['file_name']} ({info['size_mb']:.1f} MB)")
                
            try:
                idx = int(input("\nSelect checkpoint number to save (or 0 to cancel): ")) - 1
                if idx < 0:
                    continue
                    
                selected = checkpoints[idx]
                print(f"\nSelected: {os.path.basename(selected)}")
                
                # Ask for a new name
                new_name = input("Enter a name for the saved model (or press Enter for auto-name): ").strip()
                
                # Save the checkpoint
                if manager.save_checkpoint(selected, new_name):
                    print("\nCheckpoint saved successfully to models directory!")
                    
                    # Ask about loading and testing
                    test = input("Would you like to test load the model? (y/n): ").strip().lower()
                    if test == 'y':
                        model_path = os.path.join(MODELS_DIR, new_name if new_name else manager.get_checkpoint_info(selected)['file_name'])
                        model = ChessNN()  # Create default model
                        try:
                            state_dict = torch.load(model_path, map_location='cpu')
                            model.load_state_dict(state_dict)
                            print("Model loaded successfully!")
                            print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
                        except Exception as e:
                            print(f"Error testing model: {e}")
                else:
                    print("\nFailed to save the checkpoint.")
            except (ValueError, IndexError) as e:
                print(f"Invalid selection: {e}")
                
        elif choice == "3":
            confirm = input("Are you sure you want to delete all temporary checkpoint directories? (y/n): ").strip().lower()
            if confirm == 'y':
                manager.clean_temp_directories()
                
        elif choice == "4":
            print("\nExiting Checkpoint Manager.")
            break
            
        else:
            print("\nInvalid choice. Please select a number between 1 and 4.")

if __name__ == "__main__":
    main()