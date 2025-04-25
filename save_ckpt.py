import torch
import os
import sys
import traceback

try:
    from neuralNetwork import ChessNN
except ImportError:
    print("Error: Could not import ChessNN from neuralNetwork.")
    sys.exit(1)

def convert_checkpoint(ckpt_path, output_path, input_channels, **kwargs):
    """
    Loads a PyTorch Lightning checkpoint (.ckpt) and saves only the model's
    state_dict to a standard PyTorch file (.pth).

    Args:
        ckpt_path (str): Path to the input .ckpt file.
        output_path (str): Path to save the output .pth file.
        input_channels (int): The number of input channels required by ChessNN.__init__.
        **kwargs: Additional keyword arguments required by ChessNN.__init__ method
                  that are not saved in the checkpoint's hyperparameters.
                  Example: learning_rate=0.001 (if not saved in ckpt)
    """
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint file not found at '{ckpt_path}'")
        return False

    print(f"Loading model from checkpoint: {ckpt_path}")
    try:
        model = ChessNN.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            map_location='cpu',
            input_channels=input_channels,
            **kwargs
        )
        model.eval() # Model to eval
        print("Model loaded successfully.")

    except FileNotFoundError:
        print(f"Error: Checkpoint file not found: '{ckpt_path}'")
        return False
    except TypeError as e:
         print(f"Error: Loading failed. Check required arguments for ChessNN: {e}")
         return False
    except Exception as e:
        print(f"Unexpected error during loading: {e}")
        traceback.print_exc()
        return False

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir: # Create directory if needed
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating output directory '{output_dir}': {e}")
            return False

    print(f"Saving model state_dict to: {output_path}")
    try:
        # Save only the state_dict
        torch.save(model.state_dict(), output_path)
        print("Model state_dict saved successfully.")
        return True
    except Exception as e:
        print(f"Error during saving: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":

    # 1. Path to the checkpoint file
    ckpt_file_to_convert = "best-chunked-model-epoch=08-val_loss=1.5990.ckpt"

    # 2. Path for the output file
    output_weights_file = "models/converted_model_weights.pth"

    # 3. Number of input channels the ChessNN model expects
    model_input_channels = 18

    other_model_args = {
    }

    print("--- Starting Checkpoint Conversion ---")
    print(f"Input Checkpoint: {ckpt_file_to_convert}")
    print(f"Output Weights File: {output_weights_file}")
    print(f"Model Input Channels: {model_input_channels}")
    if other_model_args:
        print(f"Other Model Args: {other_model_args}")

    success = convert_checkpoint(
        ckpt_path=ckpt_file_to_convert,
        output_path=output_weights_file,
        input_channels=model_input_channels,
        **other_model_args
    )

    if success:
        print("--- Conversion Finished ---")
    else:
        print("--- Conversion Failed ---")