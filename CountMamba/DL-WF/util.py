from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn.functional as F
import numpy as np
import sys
import math
from torch.optim.lr_scheduler import LambdaLR


def gen_one_hot(arr, num_classes):
    binary = np.zeros((arr.shape[0], num_classes))
    for i in range(arr.shape[0]):
        binary[i, arr[i]] = 1

    return binary


def compute_metric(y_true_fine, y_pred_fine):
    y_true_fine = y_true_fine.reshape(-1, y_true_fine.shape[-1])
    y_pred_fine = y_pred_fine.reshape(-1, y_pred_fine.shape[-1])

    num_classes = np.max(y_true_fine) + 1
    y_true_fine = gen_one_hot(y_true_fine, num_classes)
    y_pred_fine = gen_one_hot(y_pred_fine, num_classes)

    result = measurement(y_true_fine, y_pred_fine, eval_metrics="Accuracy Precision Recall F1-score")
    return result


def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def pad_sequence(sequence, length):
    if len(sequence) >= length:
        sequence = sequence[:length]
    else:
        sequence = np.pad(sequence, (0, length - len(sequence)), "constant", constant_values=0.0)

    return sequence


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.log.flush()


def process_TAM(sequence, maximum_load_time, max_matrix_len):
    feature = np.zeros((2, max_matrix_len))  # Initialize feature matrix

    for pack in sequence:
        if pack == 0:
            break  # End of sequence
        elif pack > 0:
            if pack >= maximum_load_time:
                feature[0, -1] += 1  # Assign to the last bin if it exceeds maximum load time
            else:
                idx = int(pack * (max_matrix_len - 1) / maximum_load_time)
                feature[0, idx] += 1
        else:
            pack = np.abs(pack)
            if pack >= maximum_load_time:
                feature[1, -1] += 1  # Assign to the last bin if it exceeds maximum load time
            else:
                idx = int(pack * (max_matrix_len - 1) / maximum_load_time)
                feature[1, idx] += 1
    return feature


def measurement(y_true, y_pred, eval_metrics):
    eval_metrics = eval_metrics.split(" ")
    results = {}
    for eval_metric in eval_metrics:
        if eval_metric == "Accuracy":
            results[eval_metric] = round(accuracy_score(y_true, y_pred) * 100, 2)
        elif eval_metric == "Precision":
            results[eval_metric] = round(precision_score(y_true, y_pred, average="macro") * 100, 2)
        elif eval_metric == "Recall":
            results[eval_metric] = round(recall_score(y_true, y_pred, average="macro") * 100, 2)
        elif eval_metric == "F1-score":
            results[eval_metric] = round(f1_score(y_true, y_pred, average="macro") * 100, 2)
        else:
            raise ValueError(f"Metric {eval_metric} is not matched.")
    return results


def knn_monitor(net, device, memory_data_loader, test_data_loader, num_classes, k=200, t=0.1):
    """
    Perform k-Nearest Neighbors (kNN) monitoring.

    Parameters:
    net (nn.Module): The neural network model.
    device (torch.device): The device to run the computations on.
    memory_data_loader (DataLoader): DataLoader for the memory bank.
    test_data_loader (DataLoader): DataLoader for the test data.
    num_classes (int): Number of classes.
    k (int): Number of nearest neighbors to use.
    t (float): Temperature parameter for scaling.

    Returns:
    tuple: True labels and predicted labels.
    """
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


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    """
    Predict labels using k-Nearest Neighbors (kNN) with cosine similarity.

    Parameters:
    feature (Tensor): Feature tensor.
    feature_bank (Tensor): Feature bank tensor.
    feature_labels (Tensor): Labels corresponding to the feature bank.
    classes (int): Number of classes.
    knn_k (int): Number of nearest neighbors to use.
    knn_t (float): Temperature parameter for scaling.

    Returns:
    Tensor: Predicted labels.
    """
    feature_labels = feature_labels.long()
    
    sim_matrix = torch.mm(feature, feature_bank)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)

    return pred_labels
