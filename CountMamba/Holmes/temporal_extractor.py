import numpy as np
import os
import argparse
from tqdm import tqdm

# Argument parser for command-line options, arguments, and sub-commands
parser = argparse.ArgumentParser(description='Temporal feature extraction of Holmes')

# Define command-line arguments
parser.add_argument("--dataset", type=str, default="CW", help="Dataset name")
parser.add_argument("--in_file", type=str, default="valid", help="Input file name")

# Parse command-line arguments
args = parser.parse_args()

# Construct the input path for the dataset
in_path = os.path.join("../npz_dataset", args.dataset)

# Construct the output file path
out_file = os.path.join(in_path, f"temporal_{args.in_file}.npz")

# Load data from the specified input file
data = np.load(os.path.join(in_path, f"{args.in_file}.npz"))
X = data["X"]
y = data["y"]


def extract_temporal_feature(X, feat_length=1000):
    abs_X = np.absolute(X)
    new_X = []

    for idx in tqdm(range(X.shape[0])):
        temporal_array = np.zeros((2, feat_length))
        loading_time = abs_X[idx].max()
        interval = 1.0 * loading_time / feat_length

        for packet in X[idx]:
            if packet == 0:
                break
            elif packet > 0:
                order = int(packet / interval)
                if order >= feat_length:
                    order = feat_length - 1
                temporal_array[0][order] += 1
            else:
                order = int(-packet / interval)
                if order >= feat_length:
                    order = feat_length - 1
                temporal_array[1][order] += 1
        new_X.append(temporal_array)
    new_X = np.array(new_X)
    return new_X


# Extract temporal features from the input data
timestamp = X[:, :, 0]
sign = np.sign(X[:, :, 1])
X = timestamp * sign

temporal_X = extract_temporal_feature(X)

# Print the shape of the extracted temporal features
print("Shape of temporal_X:", temporal_X.shape)

# Save the extracted features and labels to the output file
np.savez(out_file, X=temporal_X, y=y)
