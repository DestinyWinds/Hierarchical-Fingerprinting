import argparse
import os
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="WFlib")
parser.add_argument("--dataset", type=str, default="Open_2tab", help="Dataset name")
args = parser.parse_args()

infile = os.path.join("../npz_dataset", args.dataset, "data.npz")
outfile = os.path.join("../npz_dataset", args.dataset, "data.npz")
print("loading...", infile)
data = np.load(infile)

direction = data["direction"]
time = data["time"]
label = data["label"]


def pad_sequence(sequence, length):
    if len(sequence) >= length:
        sequence = sequence[:length]
    else:
        sequence = np.pad(sequence, (0, length - len(sequence)), "constant", constant_values=0.0)

    return sequence


file_sequence = []
file_labels = []
for idx in tqdm(range(len(direction))):
    timestamp = np.trim_zeros(time[idx], "b")
    timestamp = timestamp - 1.0
    timestamp[timestamp == 0.0] = 1e-6
    timestamp = np.pad(timestamp, (0, 10000 - len(timestamp)), mode='constant', constant_values=0)
    timestamp = timestamp.reshape((1, -1, 1))

    packet_length = direction[idx].reshape((1, -1, 1))
    packet_data = np.concatenate([timestamp, packet_length], axis=-1)
    packet_data = packet_data.astype(np.float32)

    file_sequence.append(packet_data)
    file_labels.append(label[idx])

X = np.concatenate(file_sequence, axis=0)
labels = np.array(file_labels)

labels = labels.astype(np.uint8)
print(f"Train: X = {X.shape}, y = {labels.shape}")
np.savez_compressed(outfile, X=X, y=labels)
