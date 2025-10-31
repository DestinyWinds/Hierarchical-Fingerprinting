import argparse
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description="WFlib")
parser.add_argument("--dataset", type=str, default="OW", help="Dataset name")
args = parser.parse_args()

in_path = os.path.join("../dataset", args.dataset)
filenames = glob.glob(in_path + "/*")
dataset_path = os.path.join("../npz_dataset", args.dataset)
os.makedirs(dataset_path, exist_ok=True)


def pad_sequence(sequence, length):
    if len(sequence) >= length:
        sequence = sequence[:length]
    else:
        sequence = np.pad(sequence, (0, length - len(sequence)), "constant", constant_values=0.0)

    return sequence


def parse_file(files):
    file_labels = []
    file_sequence = []
    for file in tqdm(files):
        if not os.path.getsize(file) > 0:
            continue

        if "-" not in file.split("/")[-1]:
            file_labels.append(-1)
        else:
            file_labels.append(int(file.split("/")[-1].split("-")[0]))

        with open(file, 'r') as f:
            tmp = f.readlines()
        data = pd.Series(tmp).str.slice(0, -1).str.split('\t', expand=True).astype('float').to_numpy()

        timestamp = pad_sequence(data[:, 0], 10000)

        timestamp = np.trim_zeros(timestamp, "b")
        timestamp[timestamp == 0.0] = 1e-6
        timestamp = np.pad(timestamp, (0, 10000 - len(timestamp)), mode='constant', constant_values=0)

        timestamp = timestamp.reshape((1, -1, 1))
        packet_length = pad_sequence(data[:, 1], 10000).reshape((1, -1, 1))
        packet_data = np.concatenate([timestamp, packet_length], axis=-1)
        packet_data = packet_data.astype(np.float32)

        file_sequence.append(packet_data)

    return np.concatenate(file_sequence, axis=0), np.array(file_labels)


X, labels = parse_file(filenames)
max_category = np.max(labels[labels != -1])
labels[labels == -1] = max_category + 1

labels = labels.astype(np.uint8)
print(f"Train: X = {X.shape}, y = {labels.shape}")
np.savez_compressed(os.path.join(dataset_path, "data.npz"), X=X, y=labels)
