import os
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from RF_model import RF


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
ckp_path = os.path.join(args.checkpoints, args.dataset, "RF_IS")
os.makedirs(ckp_path, exist_ok=True)
out_file = os.path.join(ckp_path, f"max_f1.pth")


def load_data(data_path):
    data = np.load(data_path)
    X = data["X"]
    y = data["y"]

    X = torch.tensor(X[:, np.newaxis], dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.int64)
    return X, y


train_X, train_y = load_data(os.path.join(in_path, f"temporal_train.npz"))
valid_X, valid_y = load_data(os.path.join(in_path, f"temporal_valid.npz"))

num_classes = len(np.unique(train_y))

train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=200, shuffle=True, drop_last=True, num_workers=0)

valid_dataset = torch.utils.data.TensorDataset(valid_X, valid_y)
valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=200, shuffle=False, drop_last=False, num_workers=0)

model = RF(num_classes)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()


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

    with torch.no_grad():
        model.eval()

        valid_pred = []
        valid_true = []
        for index, cur_data in enumerate(valid_iter):
            cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
            outs = model(cur_X)

            cur_pred = torch.argsort(outs, dim=1, descending=True)[:,0]
            valid_pred.append(cur_pred.cpu().numpy())
            valid_true.append(cur_y.cpu().numpy())

        valid_pred = np.concatenate(valid_pred)
        valid_true = np.concatenate(valid_true)

        valid_result = measurement(valid_true, valid_pred)
        print(f"{epoch}: {valid_result}")

        if valid_result["F1-score"] > metric_best_value:
            metric_best_value = valid_result["F1-score"]
            torch.save(model.state_dict(), out_file)
