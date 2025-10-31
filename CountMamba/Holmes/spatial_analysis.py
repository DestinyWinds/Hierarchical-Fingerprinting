import random
import os
import torch
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity

from Holmes_model import Holmes


# Set a fixed seed for reproducibility of experiments
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# Argument parser for command-line options, arguments, and sub-commands
parser = argparse.ArgumentParser(description="Spatial analysis")
parser.add_argument("--dataset", type=str, default="CW", help="Dataset name")
parser.add_argument("--checkpoints", type=str, default="../checkpoints/", help="Directory to save model checkpoints")
args = parser.parse_args()

device = torch.device("cuda")

# Construct paths for input dataset and model checkpoints
in_path = os.path.join("../npz_dataset", args.dataset)
ckp_path = os.path.join(args.checkpoints, args.dataset, "Holmes")
out_file = os.path.join(ckp_path, "spatial_distribution.npz")


def load_data(data_path):
    data = np.load(data_path)
    X = data["X"]
    y = data["y"]

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.int64)
    return X, y


valid_X, valid_y = load_data(os.path.join(in_path, f"taf_aug_valid.npz"))
num_classes = len(np.unique(valid_y))

# Print dataset information
print(f"Valid: X={valid_X.shape}, y={valid_y.shape}")
print(f"num_classes: {num_classes}")

valid_dataset = torch.utils.data.TensorDataset(valid_X, valid_y)
valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=0)

model = Holmes(num_classes)
model.load_state_dict(torch.load(os.path.join(ckp_path, f"max_f1.pth"), map_location="cpu"))
model.to(device)

# Initialize a dictionary to store embeddings for each class
embs_pool = {}
for web in range(num_classes):
    embs_pool[web] = []

# Collect embeddings for the validation set
with torch.no_grad():
    model.eval()
    for index, cur_data in enumerate(valid_iter):
        cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
        embs = model(cur_X).cpu().numpy()
        for i, web in enumerate(cur_y.cpu().numpy()):
            embs_pool[web].append(embs[i])


def median_absolute_deviation(data):
    median = np.median(data)
    deviations = np.abs(data - median)
    mad = np.median(deviations)
    return mad


# Calculate centroids and radii for each class
webs_centroid = []
webs_radius = []
for web in range(num_classes):
    cur_embs = np.array(embs_pool[web])
    cur_centroid = cur_embs.mean(axis=0)
    webs_centroid.append(cur_centroid)

    cur_radius = 1.0 - cosine_similarity(cur_embs, cur_centroid.reshape(1, -1))
    webs_radius.append(median_absolute_deviation(cur_radius))

webs_centroid = np.array(webs_centroid)
webs_radius = np.array(webs_radius)

# Adjust radii to ensure no overlap between different classes
for web1 in range(num_classes):
    for web2 in range(web1 + 1, num_classes):
        centroid_1 = webs_centroid[web1]
        centroid_2 = webs_centroid[web2]
        distance = 1.0 - cosine_similarity(centroid_1.reshape(1, -1), centroid_2.reshape(1, -1))[0, 0]
        radius_1 = webs_radius[web1]
        radius_2 = webs_radius[web2]
        if distance <= radius_1 + radius_2:
            print(f"{web1} vs {web2}: distance = {distance}, r1 = {webs_radius[web1]}, r2 = {webs_radius[web2]}")
            diff = radius_1 + radius_2 - distance
            webs_radius[web1] -= 1.0 * diff * radius_1 / (radius_1 + radius_2)
            webs_radius[web2] -= 1.0 * diff * radius_2 / (radius_1 + radius_2)

# Print completion message and save results
print(f"Generate {out_file} done")
np.savez(out_file, centroid=webs_centroid, radius=webs_radius)
