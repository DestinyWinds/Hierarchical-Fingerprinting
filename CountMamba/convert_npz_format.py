#!/usr/bin/env python3
"""
Data Format Converter - Designed for Tor Cell Traffic
Convert the "timestamp*direction" format to the [timestamp, direction] format expected by CountMamba

Input format: X shape = (num_samples, seq_len)
         Each value = timestamp × direction (+ for send, - for receive)
Output format: X shape = (num_samples, seq_len, 2)
         X[:,:,0] = timestamp (absolute value)
         X[:,:,1] = direction (±1, fixed size Tor cell, keep only direction)

"""

import os
import argparse
import numpy as np
from tqdm import tqdm

def convert_format(input_file, output_file):
    """
    
    Parameters:
        input_file: Path to input npz file
        output_file: Path to output npz file

    """
    # Load original data
    print(f"Loading data: {input_file}")
    data = np.load(input_file)
    X_old = data['X']
    y = data['y']
    
    print(f"Original data format: X={X_old.shape}, y={y.shape}")
    print(f"Data types: X.dtype={X_old.dtype}, y.dtype={y.dtype}")
    
    # Extract timestamps and directions
    num_samples, seq_len = X_old.shape
    X_new = np.zeros((num_samples, seq_len, 2), dtype=np.float32)
    
    print("Converting format...")
    for i in tqdm(range(num_samples)):
        # Extract timestamps (absolute)
        timestamps = np.abs(X_old[i])
        X_new[i, :, 0] = timestamps
        
        # Extract directions (Tor cell: +1=send, -1=receive, 0=pad)
        directions = np.sign(X_old[i])
        X_new[i, :, 1] = directions
    
    print(f"Converted data format: X={X_new.shape}, y={y.shape}")
    print(f"Converted data types: X.dtype={X_new.dtype}, y.dtype={y.dtype}")

    # Display some statistics
    sample_idx = 0
    non_zero_mask = X_old[sample_idx] != 0
    if np.any(non_zero_mask):
        print(f"\nSample {sample_idx}'s first 10 non-zero packets:")
        print(f"  Original format (timestamp*direction): {X_old[sample_idx][non_zero_mask][:10]}")
        print(f"  Converted timestamps: {X_new[sample_idx][non_zero_mask][:10, 0]}")
        print(f"  Converted directions: {X_new[sample_idx][non_zero_mask][:10, 1]}")

    # Save converted data
    print(f"\nSaving converted data to: {output_file}")
    np.savez_compressed(output_file, X=X_new, y=y)
    print("Conversion complete!")
    
    return X_new, y


def batch_convert(input_dir, output_dir):
    """
    Batch convert all npz files in a directory (designed for Tor Cell)
    
    Parameters:
        input_dir: Input directory
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all npz files
    npz_files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]
    
    if not npz_files:
        print(f"No npz files found in {input_dir}")
        return
    
    print(f"Found {len(npz_files)} npz files")
    
    for npz_file in npz_files:
        input_path = os.path.join(input_dir, npz_file)
        output_path = os.path.join(output_dir, npz_file)
        
        print(f"\n{'='*60}")
        print(f"Processing file: {npz_file}")
        print(f"{'='*60}")
        
        try:
            convert_format(input_path, output_path)
        except Exception as e:
            print(f"Error: An error occurred while processing {npz_file}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Convert timestamp*direction format to CountMamba expected format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Convert a single file
  python convert_npz_format.py -i input.npz -o output.npz

  # Batch convert a directory
  python convert_npz_format.py -i /path/to/input_directory -o /path/to/output_directory --batch

        """
    )
    
    parser.add_argument('-i', '--input', type=str, required=True,
                       help='Input npz file or directory path')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='Output npz file or directory path')
    parser.add_argument('--batch', action='store_true',
                       help='Batch processing mode (process entire directory)')
    
    args = parser.parse_args()
    
    if args.batch:
        batch_convert(args.input, args.output)
    else:
        convert_format(args.input, args.output)


if __name__ == "__main__":
    main()

