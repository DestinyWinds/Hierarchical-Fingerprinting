import numpy as np
import random
import torch
import argparse
import os
from sklearn.metrics.pairwise import cosine_similarity
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from Holmes_model import Holmes


# Set a fixed seed for reproducibility
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# Argument parser for command-line options, arguments, and sub-commands
parser = argparse.ArgumentParser(description="WFlib")
parser.add_argument("--dataset", type=str, default="CW", help="Dataset name")
parser.add_argument("--test_file", type=str, default="taf_test_p20", help="Test file")
parser.add_argument("--result_file", type=str, default="test_p20", help="File to save test results")
parser.add_argument("--checkpoints", type=str, default="../checkpoints/", help="Location of model checkpoints")
args = parser.parse_args()

device = torch.device("cuda")

in_path = os.path.join("../npz_dataset", args.dataset)
log_path = os.path.join("../logs", args.dataset, "Holmes")
ckp_path = os.path.join(args.checkpoints, args.dataset, "Holmes")
os.makedirs(log_path, exist_ok=True)
out_file = os.path.join(log_path, f"{args.result_file}.json")


def load_data(data_path):
    data = np.load(data_path)
    X = data["X"]
    y = data["y"]

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.int64)
    return X, y


test_X, test_y = load_data(os.path.join(in_path, f"{args.test_file}.npz"))
num_classes = len(np.unique(test_y))

# Print dataset information
print(f"Test: X={test_X.shape}, y={test_y.shape}")
print(f"num_classes: {num_classes}")

test_dataset = torch.utils.data.TensorDataset(test_X, test_y)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=0)

model = Holmes(num_classes)
model.load_state_dict(torch.load(os.path.join(ckp_path, f"max_f1.pth"), map_location="cpu"))
model.to(device)

open_threshold = 1e-2
spatial_dist_file = os.path.join(ckp_path, "spatial_distribution.npz")
spatial_data = np.load(spatial_dist_file)
webs_centroid = spatial_data["centroid"]
webs_radius = spatial_data["radius"]


def measurement(y_true, y_pred):
    results = {}
    results["Accuracy"] = round(accuracy_score(y_true, y_pred), 4)
    results["Precision"] = round(precision_score(y_true, y_pred, average="macro"), 4)
    results["Recall"] = round(recall_score(y_true, y_pred, average="macro"), 4)
    results["F1-score"] = round(f1_score(y_true, y_pred, average="macro"), 4)

    return results


with torch.no_grad():
    model.eval()
    y_pred = []
    y_true = []

    for index, cur_data in enumerate(test_iter):
        cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
        embs = model(cur_X).cpu().numpy()
        cur_y = cur_y.cpu().numpy()

        all_sims = 1 - cosine_similarity(embs, webs_centroid)
        all_sims -= webs_radius
        outs = np.argmin(all_sims, axis=1)

        # if scenario == "Open-world":
        #     outs_d = np.min(all_sims, axis=1)
        #     open_indices = np.where(outs_d > open_threshold)[0]
        #     outs[open_indices] = num_classes - 1

        y_pred.append(outs)
        y_true.append(cur_y)
    y_pred = np.concatenate(y_pred).flatten()
    y_true = np.concatenate(y_true).flatten()

    result = measurement(y_true, y_pred)
    print(result)

    with open(out_file, "w") as fp:
        json.dump(result, fp, indent=4)

