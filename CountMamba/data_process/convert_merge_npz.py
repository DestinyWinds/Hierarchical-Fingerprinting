import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
import json
from itertools import chain

parser = argparse.ArgumentParser(description="WFlib")
parser.add_argument("--input_file", type=str, default="/nvme/dxw/TMWF-main/dataset/tbb_multi_tab/", help="Dataset name")
args = parser.parse_args()

outfile = os.path.join("../npz_dataset", args.input_file.strip("/").split("/")[-1], "data.npz")
os.makedirs(os.path.join("../npz_dataset", args.input_file.strip("/").split("/")[-1]), exist_ok=True)


def pad_sequence(sequence, length):
    if len(sequence) >= length:
        sequence = sequence[:length]
    else:
        sequence = np.pad(sequence, (0, length - len(sequence)), "constant", constant_values=0.0)

    return sequence


train_data = np.load(os.path.join(args.input_file, "COCO_train_0"), allow_pickle=True)
test_data = np.load(os.path.join(args.input_file, "COCO_test_0"), allow_pickle=True)

train_lb = np.load(os.path.join(args.input_file, "BAPM_train_0.npz"), allow_pickle=True)["label"]
test_lb = np.load(os.path.join(args.input_file, "BAPM_test_0.npz"), allow_pickle=True)["label"]

with open(os.path.join(args.input_file, "train_0_annotations.json"), 'r', encoding='utf-8') as file:
    train_annotation = json.load(file)["annotations"]
with open(os.path.join(args.input_file, "test_0_annotations.json"), 'r', encoding='utf-8') as file:
    test_annotation = json.load(file)["annotations"]


def extract_annotation(annotations):
    result_annotation = []
    for i in range(len(annotations) // 2):
        indice_annotation = annotations[i * 2: (i + 1) * 2]
        assert indice_annotation[0]["trace_id"] == i and indice_annotation[1]["trace_id"] == i
        bbox = [indice_annotation[0]["bbox"][0], indice_annotation[0]["bbox"][2],
                indice_annotation[1]["bbox"][0], indice_annotation[1]["bbox"][2]]
        result_annotation.append(bbox)
    return result_annotation


train_annotation = extract_annotation(train_annotation)
test_annotation = extract_annotation(test_annotation)

annotations = np.array(train_annotation + test_annotation)
direction = train_data["data"] + test_data["data"]
time = train_data["time"] + test_data["time"]
BAPM_lb = np.concatenate([train_lb, test_lb], axis=0)

direction = np.array([pad_sequence(seq, 10000) for seq in direction])
time = np.array([pad_sequence(seq, 10000) for seq in time])
num_class = np.max(BAPM_lb) + 1
label = np.zeros((BAPM_lb.shape[0], num_class))
for i, lb in enumerate(BAPM_lb):
    label[i, lb] = 1


file_sequence = []
file_labels = []
for idx in tqdm(range(len(direction))):
    timestamp = np.trim_zeros(time[idx], "b")
    timestamp[timestamp == 0.0] = 1e-6
    timestamp = np.pad(timestamp, (0, 10000 - len(timestamp)), mode='constant', constant_values=0)
    timestamp = timestamp.reshape((1, -1, 1))

    packet_length = direction[idx].reshape((1, -1, 1))
    packet_data = np.concatenate([timestamp, packet_length], axis=-1)

    file_sequence.append(packet_data)
    file_labels.append(label[idx])

X = np.concatenate(file_sequence, axis=0)
labels = np.array(file_labels)

y_true_coarse = [torch.nonzero(sample).squeeze().tolist() for sample in torch.tensor(labels)]
y_true_coarse = [element if isinstance(element, list) else [element] for element in y_true_coarse]
replace_num = max(list(chain.from_iterable(y_true_coarse))) + 1
y_true_coarse = np.array([element if len(element) == 2 else element + [replace_num] for element in y_true_coarse])

binary = np.zeros((y_true_coarse.shape[0], replace_num + 1))
for i in range(y_true_coarse.shape[0]):
    binary[i, y_true_coarse[i]] = 1

BAPM = np.concatenate([annotations, BAPM_lb], axis=1)

X = X.astype(np.float32)
binary = binary.astype(np.uint8)
print(f"Train: X = {X.shape}, y = {labels.shape}, BAPM = {BAPM.shape}")
np.savez_compressed(outfile, X=X, y=binary, BAPM=BAPM)
