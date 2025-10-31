import argparse
import configparser
import numpy as np
import random
import tqdm
import torch
import os
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import json

from dataset_util import DirectionDataset, DTDataset, DT2Dataset, RFDataset
from model import AWF, DF, VarCNN
from model_tiktok import TikTok
from model_TF import TF
from model_TMWF import TMWF
from model_RF import RF
from model_MultiTabRF import MultiTabRF
from model_ARES import Trans_WF as ARES
from util import measurement, knn_monitor, compute_metric
import warnings

warnings.filterwarnings("ignore")

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
parser.add_argument('--config', default="config/RF.ini")
parser.add_argument("--load_ratio", type=int, default=100)
parser.add_argument("--result_file", type=str, default="test_p100", help="File to save test results")
parser.add_argument("--checkpoints", type=str, default="../checkpoints/", help="Location of model checkpoints")
parser.add_argument("--num_tabs", type=int, default=1)
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.config)

# Preparation
device = torch.device("cuda")

in_path = os.path.join("../npz_dataset", args.dataset)
log_path = os.path.join("../logs", args.dataset, config['config']['model'])
ckp_path = os.path.join(args.checkpoints, args.dataset, config['config']['model'])
os.makedirs(log_path, exist_ok=True)
out_file = os.path.join(log_path, f"{args.result_file}.json")


# Dataset
def load_data(data_path):
    data = np.load(data_path)
    X = data["X"]
    y = data["y"]

    return X, y


valid_X, valid_y = load_data(os.path.join("../npz_dataset", args.dataset, f"valid.npz"))
if args.load_ratio != 100:
    test_X, test_y = load_data(os.path.join(in_path, f"test_p{args.load_ratio}.npz"))
else:
    test_X, test_y = load_data(os.path.join(in_path, f"test.npz"))
if args.num_tabs == 1:
    num_classes = len(np.unique(test_y))
else:
    num_classes = test_y.shape[1]

if config['config']['model'] in ["AWF", "DF", "TF", "TMWF", "ARES"]:
    val_set = DirectionDataset(valid_X, valid_y, int(config['config']['seq_len']))
    test_dataset = DirectionDataset(test_X, test_y, int(config['config']['seq_len']))
elif config['config']['model'] in ["TikTok"]:
    val_set = DTDataset(valid_X, valid_y, int(config['config']['seq_len']))
    test_dataset = DTDataset(test_X, test_y, int(config['config']['seq_len']))
elif config['config']['model'] in ["VarCNN"]:
    val_set = DT2Dataset(valid_X, valid_y, int(config['config']['seq_len']))
    test_dataset = DT2Dataset(test_X, test_y, int(config['config']['seq_len']))
elif config['config']['model'] in ["RF", "MultiTabRF"]:
    val_set = RFDataset(valid_X, valid_y, int(config['config']['seq_len']))
    test_dataset = RFDataset(test_X, test_y, int(config['config']['seq_len']))

val_loader = DataLoader(val_set, batch_size=int(config['config']['batch_size']),
                        shuffle=False, drop_last=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=int(config['config']['batch_size']),
                         shuffle=False, drop_last=False, num_workers=4)

# Model
if config['config']['model'] == "AWF":
    model = AWF(num_classes=num_classes)
elif config['config']['model'] == "DF":
    model = DF(num_classes=num_classes)
elif config['config']['model'] == "TikTok":
    model = TikTok(num_classes=num_classes)
elif config['config']['model'] == "VarCNN":
    model = VarCNN(num_classes=num_classes)
elif config['config']['model'] == "TF":
    model = TF(num_classes=num_classes)
elif config['config']['model'] == "TMWF":
    model = TMWF(num_classes=num_classes)
elif config['config']['model'] == "ARES":
    model = ARES(num_classes=num_classes)
elif config['config']['model'] == "RF":
    model = RF(num_classes=num_classes)
elif config['config']['model'] == "MultiTabRF":
    model = MultiTabRF(num_classes=num_classes)

model.load_state_dict(torch.load(os.path.join(ckp_path, f"max_f1.pth"), map_location="cpu"))
model.to(device)

if config['config']['model'] == "TF":
    valid_true, valid_pred = knn_monitor(model, device, val_loader, test_loader, num_classes, 10)
    result = measurement(valid_true, valid_pred, config['config']['eval_metrics'])
else:
    if args.num_tabs > 1:
        y_pred_score = np.zeros((0, num_classes))
        y_true = np.zeros((0, num_classes))
        with torch.no_grad():
            model.eval()
            for index, cur_data in enumerate(tqdm.tqdm(test_loader)):
                cur_X, cur_y = cur_data[0].cuda(), cur_data[1].cuda()
                outs = model(cur_X)
                y_pred_score = np.append(y_pred_score, outs.cpu().numpy(), axis=0)
                y_true = np.append(y_true, cur_y.cpu().numpy(), axis=0)

            max_tab = 5
            tp = {}
            for tab in range(1, max_tab + 1):
                tp[tab] = 0

            for idx in range(y_pred_score.shape[0]):
                cur_pred = y_pred_score[idx]
                for tab in range(1, max_tab + 1):
                    target_webs = cur_pred.argsort()[-tab:]
                    for target_web in target_webs:
                        if y_true[idx, target_web] > 0:
                            tp[tab] += 1
            mapk = .0
            for tab in range(1, max_tab + 1):
                p_tab = tp[tab] / (y_true.shape[0] * tab)
                mapk += p_tab
                if tab == args.num_tabs:
                    result = {
                        f"p@{tab}": round(p_tab, 4) * 100,
                        f"ap@{tab}": round(mapk / tab, 4) * 100
                    }

            y_pred_coarse = y_pred_score.argsort()[:, -2:]
            y_true_coarse = [torch.nonzero(sample).squeeze().tolist() for sample in torch.tensor(y_true)]
            y_true_coarse = np.array(y_true_coarse)

            metric_coarse_result = compute_metric(y_true_coarse, y_pred_coarse)
            result.update(metric_coarse_result)
    else:
        with torch.no_grad():
            model.eval()
            valid_pred = []
            valid_true = []

            for index, cur_data in enumerate(tqdm.tqdm(test_loader)):
                cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)

                outs = model(cur_X)
                outs = torch.argsort(outs, dim=1, descending=True)[:, 0]
                valid_pred.append(outs.cpu().numpy())
                valid_true.append(cur_y.cpu().numpy())

            valid_pred = np.concatenate(valid_pred)
            valid_true = np.concatenate(valid_true)

        result = measurement(valid_true, valid_pred, config['config']['eval_metrics'])

print(result)
with open(out_file, "w") as fp:
    json.dump(result, fp, indent=4)
