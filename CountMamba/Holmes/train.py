import os
import torch
import random
import argparse
import numpy as np
from pytorch_metric_learning import losses
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from Holmes_model import Holmes


# Set a fixed seed for reproducibility
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description="WFlib")
parser.add_argument("--dataset", type=str, default="CW", help="Dataset name")
parser.add_argument("--train_epochs", type=int, default=30, help="Train epochs")
parser.add_argument("--checkpoints", type=str, default="../checkpoints/", help="Location of model checkpoints")
args = parser.parse_args()

device = torch.device("cuda")

in_path = os.path.join("../npz_dataset", args.dataset)
ckp_path = os.path.join(args.checkpoints, args.dataset, "Holmes")
os.makedirs(ckp_path, exist_ok=True)

out_file = os.path.join(ckp_path, f"max_f1.pth")


def load_data(data_path):
    data = np.load(data_path)
    X = data["X"]
    y = data["y"]

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.int64)
    return X, y


train_X, train_y = load_data(os.path.join(in_path, f"taf_aug_train.npz"))
valid_X, valid_y = load_data(os.path.join(in_path, f"taf_aug_valid.npz"))

num_classes = len(np.unique(train_y))

# Print dataset information
print(f"Train: X={train_X.shape}, y={train_y.shape}")
print(f"Valid: X={valid_X.shape}, y={valid_y.shape}")
print(f"num_classes: {num_classes}")

train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True, num_workers=0)

valid_dataset = torch.utils.data.TensorDataset(valid_X, valid_y)
valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=0)

model = Holmes(num_classes)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

criterion = losses.SupConLoss(temperature=0.1)


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    sim_matrix = torch.mm(feature, feature_bank)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)

    return pred_labels


def knn_monitor(net, device, memory_data_loader, test_data_loader, num_classes, k=200, t=0.1):
    net.eval()
    total_num = 0
    feature_bank, feature_labels = [], []
    y_pred = []
    y_true = []

    with torch.no_grad():
        # Generate feature bank
        for data, target in memory_data_loader:
            feature = net(data.to(device))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            feature_labels.append(target)

        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous().to(device)
        feature_labels = torch.cat(feature_labels, dim=0).t().contiguous().to(device)

        # Loop through test data to predict the label by weighted kNN search
        for data, target in test_data_loader:
            data, target = data.to(device), target.to(device)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            pred_labels = knn_predict(feature, feature_bank, feature_labels, num_classes, k, t)
            total_num += data.size(0)
            y_pred.append(pred_labels[:, 0].cpu().numpy())
            y_true.append(target.cpu().numpy())

    y_true = np.concatenate(y_true).flatten()
    y_pred = np.concatenate(y_pred).flatten()

    return y_true, y_pred


def measurement(y_true, y_pred):
    results = {}
    results["Accuracy"] = round(accuracy_score(y_true, y_pred), 4)
    results["Precision"] = round(precision_score(y_true, y_pred, average="macro"), 4)
    results["Recall"] = round(recall_score(y_true, y_pred, average="macro"), 4)
    results["F1-score"] = round(f1_score(y_true, y_pred, average="macro"), 4)

    return results


metric_best_value = 0
for epoch in range(args.train_epochs):
    model.train()
    sum_loss = 0
    sum_count = 0

    for index, cur_data in tqdm(enumerate(train_iter)):
        cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
        optimizer.zero_grad()
        outs = model(cur_X)

        loss = criterion(outs, cur_y)

        loss.backward()
        optimizer.step()
        sum_loss += loss.data.cpu().numpy() * outs.shape[0]
        sum_count += outs.shape[0]

    train_loss = round(sum_loss / sum_count, 3)
    print(f"epoch {epoch}: train_loss = {train_loss}")

    valid_true, valid_pred = knn_monitor(model, device, train_iter, valid_iter, num_classes, 10)

    valid_result = measurement(valid_true, valid_pred)
    print(f"{epoch}: {valid_result}")

    if valid_result["F1-score"] > metric_best_value:
        metric_best_value = valid_result["F1-score"]
        torch.save(model.state_dict(), out_file)
