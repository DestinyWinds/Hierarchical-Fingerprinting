import argparse
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import joblib
import os
import itertools
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set a fixed seed for reproducibility
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
torch.cuda.manual_seed(fix_seed)
np.random.seed(fix_seed)
rng = np.random.RandomState(fix_seed)
cudnn.benchmark = False
cudnn.deterministic = True

# Config
parser = argparse.ArgumentParser(description="WFlib")
parser.add_argument('--dataset', default="CW")
args = parser.parse_args()
print(args)

# Dataset
ckp_path = os.path.join("../checkpoints", args.dataset, "CUMUL")
os.makedirs(ckp_path, exist_ok=True)
out_file = os.path.join(ckp_path, f"max_f1.pth")


# Dataset
def load_data(data_path):
    data = np.load(data_path)
    X = data["X"]
    y = data["y"]

    return X, y


train_X, train_y = load_data(os.path.join("../npz_dataset", args.dataset, f"train.npz"))
valid_X, valid_y = load_data(os.path.join("../npz_dataset", args.dataset, f"valid.npz"))
num_classes = len(np.unique(train_y))


def extract_feauture(sinste):  # sinste: list of packet sizes

    # first 4 features
    features = []

    total = []
    cum = []
    pos = []
    neg = []
    inSize = 0
    outSize = 0
    inCount = 0
    outCount = 0

    for i in range(0, len(sinste)):
        # incoming packets
        if sinste[i] > 0:
            inSize += sinste[i]
            inCount += 1
            # cumulated packetsizes
            if len(cum) == 0:
                cum.append(sinste[i])
                total.append(sinste[i])
                pos.append(sinste[i])
                neg.append(0)
            else:
                cum.append(cum[-1] + sinste[i])
                total.append(total[-1] + abs(sinste[i]))
                pos.append(pos[-1] + sinste[i])
                neg.append(neg[-1] + 0)

        # outgoing packets
        if sinste[i] < 0:
            outSize += abs(sinste[i])
            outCount += 1
            if len(cum) == 0:
                cum.append(sinste[i])
                total.append(abs(sinste[i]))
                pos.append(0)
                neg.append(abs(sinste[i]))
            else:
                cum.append(cum[-1] + sinste[i])
                total.append(total[-1] + abs(sinste[i]))
                pos.append(pos[-1] + 0)
                neg.append(neg[-1] + abs(sinste[i]))

    # add feature
    features.append(inCount)
    features.append(outCount)
    features.append(outSize)
    features.append(inSize)
    cumFeatures = np.interp(np.linspace(total[0], total[-1], 101), total, cum)
    for el in itertools.islice(cumFeatures, 1, None):
        features.append(el)
    return features


def extract(X):
    features = []
    for data in tqdm(X):
        packet_length = data[:, 1].astype(np.int64)
        feature = extract_feauture(packet_length)
        features.append(feature)
    features = np.array(features)
    return features


train_features = extract(train_X)
val_features = extract(valid_X)

scaler = MinMaxScaler(feature_range=(-1, 1))
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)

clf = svm.SVC(kernel='rbf')
param_grid = {'C': [2048, 4096, 8192, 16384, 32768, 65536, 131072], 'gamma': [0.125, 0.25, 0.5, 1, 2, 4, 8]}

grid_search = GridSearchCV(clf, param_grid, n_jobs=8, verbose=1)
grid_search.fit(train_features, train_y)
best_parameters = grid_search.best_estimator_.get_params()
print("Best parameters: ", best_parameters)

svm_model = svm.SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'])
svm_model.fit(train_features, train_y)
joblib.dump(svm_model, out_file)
y_pred = svm_model.predict(val_features)


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


test_result = measurement(valid_y, y_pred, "Accuracy Precision Recall F1-score")
print(test_result)
