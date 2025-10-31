import os
import argparse
import numpy as np
from tqdm import tqdm
import random


parser = argparse.ArgumentParser(description="WFlib")
parser.add_argument("--dataset", type=str, default="CW", help="Dataset name")
parser.add_argument("--in_file", type=str, default="valid", help="Input file name")
parser.add_argument("--checkpoints", type=str, default="../checkpoints/", help="Directory to save model checkpoints")
args = parser.parse_args()

# Set a fixed seed for reproducibility
fix_seed = 2024
random.seed(fix_seed)
np.random.seed(fix_seed)

in_path = os.path.join("../npz_dataset", args.dataset)
data = np.load(os.path.join(in_path, f"{args.in_file}.npz"))

temporal_data = np.load(os.path.join(args.checkpoints, args.dataset, "RF_IS", f"attr_DeepLiftShap.npz"))["attr_values"]


# Calculate effective ranges for each class based on the temporal attribution data
effective_ranges = {}
for web in range(temporal_data.shape[0]):
    cur_temporal = np.cumsum(temporal_data[web])
    cur_temporal /= cur_temporal.max()
    cur_lower = np.searchsorted(cur_temporal, 0.3, side="right") * 100 // temporal_data.shape[1]
    cur_upper = np.searchsorted(cur_temporal, 0.6, side="right") * 100 // temporal_data.shape[1]
    effective_ranges[web] = (cur_lower, cur_upper)

# Construct the output file path for the augmented data
out_file = os.path.join(in_path, f"aug_{args.in_file}.npz")


def gen_augment(data, num_aug, effective_ranges, out_file):
    X = data["X"]
    y = data["y"]

    new_X = []
    new_y = []
    abs_X = np.absolute(X)
    feat_length = X.shape[1]

    # Loop through each sample in the dataset
    for index in tqdm(range(abs_X.shape[0])):
        cur_abs_X = abs_X[index, :, 0]
        cur_web = y[index]
        loading_time = cur_abs_X.max()

        # Generate augmentations for each sample
        for ii in range(num_aug):
            p = np.random.randint(effective_ranges[cur_web][0], effective_ranges[cur_web][1])
            threshold = loading_time * p / 100

            valid_X = np.trim_zeros(cur_abs_X, "b")
            valid_index = np.where(valid_X <= threshold)

            valid_X = X[index, valid_index, :]
            valid_length = valid_X.shape[1]

            pad_width = ((0, 0), (0, feat_length - valid_length), (0, 0))
            new_X.append(
                np.pad(valid_X, pad_width, "constant", constant_values=(0, 0)))
            new_y.append(cur_web)

        # Add the original sample
        new_X.append(X[index, None])
        new_y.append(cur_web)

    new_X = np.concatenate(new_X, axis=0)
    new_y = np.array(new_y)

    # Save the augmented data to the specified output file
    np.savez(out_file, X=new_X, y=new_y)
    print(f"Generate {out_file} done.")


gen_augment(data, 2, effective_ranges, out_file)
