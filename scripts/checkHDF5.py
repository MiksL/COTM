import h5py
import hdf5plugin
import numpy as np
import os
import sys

def inspect_hdf5_file(file_path: str):
    """
    Inspects an HDF5 file, printing metadata and details about its datasets.

    Args:
        file_path: The path to the HDF5 file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return
    
    print(f"\n--- Inspecting HDF5 File: {file_path} ---")
    
    try:
        with h5py.File(file_path, 'r') as hf:
            # Print top-level attributes (metadata of the file itself)
            print("\nFile-Level Metadata (Attributes):")
            if not hf.attrs:
                print("  No file-level attributes found.")
            for key, value in hf.attrs.items():
                print(f"  - {key}: {value}")
            
            print("\n--- Datasets Information ---")
            if not hf.keys():
                print("  No datasets found in this HDF5 file.")
                return

            for dataset_name in hf.keys():
                dataset = hf[dataset_name]
                print(f"\nDataset: '{dataset_name}'")
                print(f"  Shape: {dataset.shape}")
                print(f"  Data Type (dtype): {dataset.dtype}")
                
                # Print attributes of the dataset, if any
                if dataset.attrs:
                    print("  Dataset Attributes:")
                    for attr_key, attr_val in dataset.attrs.items():
                        print(f"    - {attr_key}: {attr_val}")
                
                # Specific checks for a dataset named 'positions' (common in this project)
                if dataset_name == 'positions' and len(dataset.shape) == 4:
                    # Assuming shape is (N, C, H, W)
                    print(f"  Detected 'positions' dataset with shape (N, C, H, W): {dataset.shape}")
                    num_samples, channels, height, width = dataset.shape
                    print(f"    Number of samples (N): {num_samples}")
                    print(f"    Channels (C): {channels}")
                    print(f"    Height (H): {height}")
                    print(f"    Width (W): {width}")
                    # Add any specific warnings or info based on expected channel count, etc.
                    # if channels == 18: 
                    #     print("    Note: This dataset has 18 channels.")
                
                # Print a very small sample of the data for preview
                if dataset.size > 0: # Check if dataset is not empty
                    try:
                        # Take a small, manageable slice for preview
                        if dataset.ndim == 1:
                            sample_preview = dataset[0:min(5, dataset.shape[0])]
                        elif dataset.ndim > 1:
                            # For multi-dimensional, take the first element along the first axis, then slice inner parts
                            slicing = tuple([0] + [slice(None, min(3, dataset.shape[i])) for i in range(1, dataset.ndim)])
                            sample_preview = dataset[slicing]
                        else: # 0-dimensional array (scalar)
                            sample_preview = dataset[()]
                        
                        print(f"  Sample Preview (first few elements/slice):")
                        print(f"    {sample_preview}")
                    except Exception as sample_e:
                        print(f"    Could not retrieve sample preview: {sample_e}")
            
    except Exception as e:
        print(f"Error reading or processing HDF5 file '{file_path}': {e}")
        # import traceback # For debugging
        # traceback.print_exc()

    print("\n-----------------------------------------")

def main():
    """Main function to handle script arguments and call the inspector."""
    if len(sys.argv) < 2:
        print("Usage: python checkHDF5.py <path_to_hdf5_file>")
        print("Example: python scripts/checkHDF5.py your_data.hdf5")
        return
    
    file_to_inspect = sys.argv[1]
    inspect_hdf5_file(file_to_inspect)

if __name__ == "__main__":
    main() 