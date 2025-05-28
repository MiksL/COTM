import os
import sys
from dotenv import load_dotenv

# Ensure project root is in sys.path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from existing_games.hdfDataset import ChunkedHDF5Dataset
except ImportError as e:
    print(f"Error: Could not import ChunkedHDF5Dataset: {e}")
    print(f"Ensure this script is in a 'scripts' subdirectory and your project modules are accessible.")
    print(f"Attempted to add '{project_root}' to sys.path.")
    sys.exit(1)

def view_hdf5_file_stats():
    """Displays statistics for a user-selected HDF5 file from the GAMES_PATH directory."""
    load_dotenv() 
    games_path_env = os.getenv("GAMES_PATH")

    if not games_path_env:
        print("Error: GAMES_PATH environment variable is not set in .env file.")
        return
            
    if not os.path.isdir(games_path_env):
        print(f"Error: GAMES_PATH '{games_path_env}' is not a valid directory.")
        return

    print(f"--- HDF5 File Inspector ---")
    print(f"Searching for HDF5 files in: {games_path_env}\n")

    available_hdf5_files = sorted([
        f for f in os.listdir(games_path_env) 
        if f.endswith(".hdf5") and os.path.isfile(os.path.join(games_path_env, f))
    ])

    if not available_hdf5_files:
        print(f"No .hdf5 files found in '{games_path_env}'.")
        return

    print("Available HDF5 files:")
    for i, fname in enumerate(available_hdf5_files):
        print(f"  {i+1}. {fname}")
    
    try:
        choice_input = input("\nEnter the number of the HDF5 file to inspect: ")
        file_choice_idx = int(choice_input) - 1
        if not 0 <= file_choice_idx < len(available_hdf5_files):
            print("Invalid selection number.")
            return
    except ValueError:
        print("Invalid input. Please enter a number corresponding to the file.")
        return

    selected_file_name = available_hdf5_files[file_choice_idx]
    selected_file_path = os.path.join(games_path_env, selected_file_name)

    print(f"\n--- Statistics for: {selected_file_name} ---")
    try:
        dataset = ChunkedHDF5Dataset(data_source=selected_file_path)
        print(f"  Total Samples (Board Positions): {dataset.total_samples:,}") # Access via property
        print(f"  Samples per Chunk:             {dataset.samples_per_chunk:,}")
        print(f"  Total Number of Chunks:        {len(dataset):,}")
        if dataset.position_chunk_shape:
            print(f"  Position Tensor Shape per Chunk: {dataset.position_chunk_shape} (N, C, H, W)")
        else:
            print("  Position Chunk Shape: Not available/determined.")
        # Add any other relevant stats you want to display from the dataset object
            
    except Exception as e:
        print(f"Error reading or processing HDF5 file '{selected_file_name}': {e}")
        # import traceback # Uncomment if detailed traceback is needed for debugging
        # traceback.print_exc()

    print("\n-----------------------------")

if __name__ == "__main__":
    view_hdf5_file_stats() 