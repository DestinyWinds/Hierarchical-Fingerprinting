import numpy as np
import os
import argparse
import random
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# Set a fixed seed for reproducibility
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# Argument parser for command-line options, arguments, and sub-commands
parser = argparse.ArgumentParser(description='Feature extraction')
parser.add_argument("--dataset", type=str, default="CW", help="Dataset name")
parser.add_argument("--in_file", type=str, default="test_p10", help="input file")
args = parser.parse_args()

in_path = os.path.join("../npz_dataset", args.dataset)
out_file = os.path.join(in_path, f"taf_{args.in_file}.npz")

data = np.load(os.path.join(in_path, f"{args.in_file}.npz"))
X = data["X"]
y = data["y"]


def fast_count_burst(arr):
    diff = np.diff(arr)
    change_indices = np.nonzero(diff)[0]
    segment_starts = np.insert(change_indices + 1, 0, 0)
    segment_ends = np.append(change_indices, len(arr) - 1)
    segment_lengths = segment_ends - segment_starts + 1
    segment_signs = np.sign(arr[segment_starts])
    adjusted_lengths = segment_lengths * segment_signs

    return adjusted_lengths


def agg_interval(packets):
    features = []
    features.append([np.sum(packets > 0), np.sum(packets < 0)])

    dirs = np.sign(packets)
    assert not np.any(dir == 0), "Array contains zero!"
    bursts = fast_count_burst(dirs)
    features.append([np.sum(bursts > 0), np.sum(bursts < 0)])

    pos_bursts = bursts[bursts > 0]
    neg_bursts = np.abs(bursts[bursts < 0])
    vals = []
    if len(pos_bursts) == 0:
        vals.append(0)
    else:
        vals.append(np.mean(pos_bursts))
    if len(neg_bursts) == 0:
        vals.append(0)
    else:
        vals.append(np.mean(neg_bursts))
    features.append(vals)

    return np.array(features, dtype=np.float32)


def process_TAF(index, sequence, interval, max_len):
    timestamp = sequence[:, 0]
    sign = np.sign(sequence[:, 1])
    sequence = timestamp * sign

    packets = np.trim_zeros(sequence, "fb")
    abs_packets = np.abs(packets)
    st_time = abs_packets[0]
    st_pos = 0
    TAF = np.zeros((3, 2, max_len))

    for interval_idx in range(max_len):
        ed_time = (interval_idx + 1) * interval
        if interval_idx == max_len - 1:
            ed_pos = abs_packets.shape[0]
        else:
            ed_pos = np.searchsorted(abs_packets, st_time + ed_time)

        assert ed_pos >= st_pos, f"{index}: st:{st_pos} -> ed:{ed_pos}"
        if st_pos < ed_pos:
            cur_packets = packets[st_pos:ed_pos]
            TAF[:, :, interval_idx] = agg_interval(cur_packets)
        st_pos = ed_pos

    return index, TAF


def extract_TAF(sequences, num_workers=30):
    interval = 40
    max_len = 2000
    sequences *= 1000
    num_sequences = sequences.shape[0]
    TAF = np.zeros((num_sequences, 3, 2, max_len))

    with ProcessPoolExecutor(max_workers=min(num_workers, num_sequences)) as executor:
        futures = [executor.submit(process_TAF, index, sequences[index], interval, max_len) for index in
                   range(num_sequences)]
        with tqdm(total=num_sequences) as pbar:
            for future in as_completed(futures):
                index, result = future.result()
                TAF[index] = result
                pbar.update(1)

    return TAF


# Extract the TAF
X = extract_TAF(X)
# Print processing information
print(f"{args.in_file} process done: X = {X.shape}, y = {y.shape}")
# Save the processed data into a new .npz file
np.savez(out_file, X=X, y=y)
