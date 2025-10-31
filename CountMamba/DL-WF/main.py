import argparse
import configparser
import numpy as np
import random
import tqdm
import torch
import os
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from pytorch_metric_learning import miners, losses

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
parser.add_argument("--train_epochs", type=int, default=100, help="Train epochs")
parser.add_argument('--config', default="config/RF.ini")
parser.add_argument("--num_tabs", type=int, default=1)
args = parser.parse_args()
print(args)

config = configparser.ConfigParser()
config.read(args.config)

# Preparation
device = torch.device(config['config']['device'])

ckp_path = os.path.join("../checkpoints", args.dataset, config['config']['model'])
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
if args.num_tabs == 1:
    num_classes = len(np.unique(train_y))
else:
    num_classes = train_y.shape[1]

if config['config']['model'] in ["AWF", "DF", "TF", "TMWF", "ARES"]:
    train_set = DirectionDataset(train_X, train_y, int(config['config']['seq_len']))
    val_set = DirectionDataset(valid_X, valid_y, int(config['config']['seq_len']))
elif config['config']['model'] in ["TikTok"]:
    train_set = DTDataset(train_X, train_y, int(config['config']['seq_len']))
    val_set = DTDataset(valid_X, valid_y, int(config['config']['seq_len']))
elif config['config']['model'] in ["VarCNN"]:
    train_set = DT2Dataset(train_X, train_y, int(config['config']['seq_len']))
    val_set = DT2Dataset(valid_X, valid_y, int(config['config']['seq_len']))
elif config['config']['model'] in ["RF", "MultiTabRF"]:
    train_set = RFDataset(train_X, train_y, int(config['config']['seq_len']))
    val_set = RFDataset(valid_X, valid_y, int(config['config']['seq_len']))

train_loader = DataLoader(train_set, batch_size=int(config['config']['batch_size']),
                          shuffle=True, drop_last=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=int(config['config']['batch_size']),
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

model.to(device)
optimizer = eval(f"torch.optim.{config['config']['optimizer']}")(model.parameters(),
                                                                 lr=float(config['config']['learning_rate']))

if config['config']['model'] == "TF":
    criterion = losses.TripletMarginLoss(margin=0.1)
    miner = miners.TripletMarginMiner(margin=0.1, type_of_triplets="semihard")
elif args.num_tabs > 1:
    criterion = torch.nn.MultiLabelSoftMarginLoss()
else:
    criterion = torch.nn.CrossEntropyLoss()

# Train
metric_best_value = 0
for epoch in range(args.train_epochs):
    model.train()
    sum_loss = 0
    sum_count = 0
    for index, cur_data in enumerate(tqdm.tqdm(train_loader)):
        cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
        optimizer.zero_grad()
        outs = model(cur_X)

        if config['config']['model'] == "TF":
            hard_pairs = miner(outs, cur_y)
            loss = criterion(outs, cur_y, hard_pairs)
        else:
            loss = criterion(outs, cur_y)

        loss.backward()
        optimizer.step()
        sum_loss += loss.data.cpu().numpy() * outs.shape[0]
        sum_count += outs.shape[0]

    train_loss = round(sum_loss / sum_count, 3)
    print(f"epoch {epoch}: train_loss = {train_loss}")

    if args.num_tabs > 1:
        y_pred_score = np.zeros((0, num_classes))
        y_true = np.zeros((0, num_classes))
        with torch.no_grad():
            model.eval()
            for index, cur_data in enumerate(val_loader):
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
                print(f"p@{tab}", round(p_tab, 4) * 100, epoch)
                print(f"ap@{tab}", round(mapk / tab, 4) * 100, epoch)

                if tab == args.num_tabs:
                    ap_metric = round(mapk / tab, 4) * 100
                    if ap_metric > metric_best_value:
                        metric_best_value = ap_metric
                        torch.save(model.state_dict(), out_file)

            y_pred_coarse = y_pred_score.argsort()[:, -2:]
            y_true_coarse = [torch.nonzero(sample).squeeze().tolist() for sample in torch.tensor(y_true)]
            y_true_coarse = np.array(y_true_coarse)

            metric_coarse_result = compute_metric(y_true_coarse, y_pred_coarse)
            print(metric_coarse_result)
    else:
        if config['config']['model'] == "TF":
            valid_true, valid_pred = knn_monitor(model, device, train_loader, val_loader, num_classes, 10)
        else:
            with torch.no_grad():
                model.eval()
                valid_pred = []
                valid_true = []

                for index, cur_data in enumerate(tqdm.tqdm(val_loader)):
                    cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)

                    outs = model(cur_X)
                    if args.num_tabs == 1:
                        cur_pred = torch.argsort(outs, dim=1, descending=True)[:, 0]
                    else:
                        cur_indices = torch.argmax(outs, dim=-1).cpu()
                        cur_pred = torch.zeros((cur_indices.shape[0], num_classes))
                        for cur_tab in range(cur_indices.shape[1]):
                            row_indices = torch.arange(cur_pred.shape[0])
                            cur_pred[row_indices, cur_indices[:, cur_tab]] = 1
                    valid_pred.append(cur_pred.cpu().numpy())
                    valid_true.append(cur_y.cpu().numpy())

                valid_pred = np.concatenate(valid_pred)
                valid_true = np.concatenate(valid_true)

        valid_result = measurement(valid_true, valid_pred, config['config']['eval_metrics'])
        print(f"{epoch}: {valid_result}")

        if valid_result["F1-score"] > metric_best_value:
            metric_best_value = valid_result["F1-score"]
            torch.save(model.state_dict(), out_file)
