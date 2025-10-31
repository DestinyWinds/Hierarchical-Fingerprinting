import numpy as np
import os
import random
import argparse
from tqdm import tqdm


# Set a fixed seed for reproducibility
fix_seed = 2024
random.seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description="WFlib")
parser.add_argument("--dataset", type=str, default="Closed_2tab", help="Dataset name")
args = parser.parse_args()

in_path = os.path.join("../npz_dataset", f"{args.dataset}")
in_file = os.path.join(in_path, "test.npz")

data = np.load(in_file)
X = data["X"]
y = data["y"]
feat_length = X.shape[1]

for p in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    out_file = os.path.join(in_path, f"test_p{p}.npz")
    print(f"Generating the page loaded {p}% of traffic")

    cur_X = []
    cur_y = []
    abs_X = X[:, :, 0]

    for idx in tqdm(range(X.shape[0])):
        tmp_X = abs_X[idx]
        loading_time = tmp_X.max()
        threshold = loading_time * p / 100

        tmp_X = np.trim_zeros(tmp_X, "b")
        index = np.where(tmp_X <= threshold)

        tmp_X = X[idx, index, :]
        tmp_size = tmp_X.shape[1]

        pad_width = ((0, 0), (0, feat_length - tmp_size), (0, 0))
        tmp_X = np.pad(tmp_X, pad_width, "constant", constant_values=(0, 0))
        cur_X.append(tmp_X)
        cur_y.append(y[idx])

    cur_X = np.concatenate(cur_X, axis=0)
    cur_y = np.array(cur_y)
    print(f"Shape: X = {cur_X.shape}, y = {cur_y.shape}")
    np.savez(out_file, X=cur_X, y=cur_y)
