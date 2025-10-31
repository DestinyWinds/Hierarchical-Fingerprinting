import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
import time
import json
from sklearn.metrics.pairwise import cosine_similarity

from util import IncrementalMeanCalculator, pad_sequence, process_TAF, process_TAM
from model_CountMamba import CountMambaModel
from model_RF import RF
from model import AWF, DF, VarCNN
from model_TMWF import TMWF
from model_tiktok import TikTok
from model_Holmes import Holmes


class InferenceParams:
    def __init__(self):
        self.key_value_memory_dict = {}


parser = argparse.ArgumentParser(description="WFlib")
parser.add_argument('--dataset', default="CW")
parser.add_argument('--model', default="CountMamba", type=str)
parser.add_argument('--threshold', default=1.0, type=float)
parser.add_argument('--device', default="cpu", type=str)

# CountMamba-Model
parser.add_argument('--drop_path_rate', default=0.2, type=float)
parser.add_argument('--depth', default=3, type=int)
parser.add_argument('--embed_dim', default=64, type=int)
# CountMamba-Representation
parser.add_argument('--seq_len', default=5000, type=int)
parser.add_argument('--maximum_load_time', default=80, type=int)
parser.add_argument('--max_matrix_len', default=1800, type=int)
parser.add_argument('--time_interval_threshold', default=0.1, type=float)
parser.add_argument('--maximum_cell_number', default=2, type=int)
parser.add_argument('--log_transform', default=True)

args = parser.parse_args()

device = torch.device(args.device)

in_path = os.path.join("../npz_dataset", args.dataset)
ckp_path = os.path.join("../checkpoints/", args.dataset, args.model)
log_path = os.path.join("../logs", args.dataset, args.model)
os.makedirs(log_path, exist_ok=True)
out_file = os.path.join(log_path, f"EarlyStage_{args.threshold}_{args.device}.json")


def load_data(data_path):
    data = np.load(data_path)
    X = data["X"]
    y = data["y"]

    return X, y


test_X, test_y = load_data(os.path.join(in_path, f"test.npz"))
if args.model != "TMWF":
    test_X = test_X[:, :5000, :]
num_classes = len(np.unique(test_y))

# model
if args.model == "CountMamba":
    num_patches = args.max_matrix_len // 6
    patch_size = 2 * (args.maximum_cell_number + 1) + 2
    model = CountMambaModel(num_classes=num_classes, drop_path_rate=args.drop_path_rate, depth=args.depth,
                            embed_dim=args.embed_dim, num_patches=num_patches, patch_size=patch_size)
elif args.model == "RF":
    model = RF(num_classes=num_classes)
elif args.model == "AWF":
    model = AWF(num_classes=num_classes)
elif args.model == "DF":
    model = DF(num_classes=num_classes)
elif args.model == "TMWF":
    model = TMWF(num_classes=num_classes)
elif args.model == "TikTok":
    model = TikTok(num_classes=num_classes)
elif args.model == "VarCNN":
    model = VarCNN(num_classes=num_classes)
elif args.model == "Holmes":
    model = Holmes(num_classes=num_classes)

model.load_state_dict(torch.load(os.path.join(ckp_path, f"max_f1.pth"), map_location="cpu"))
model.to(device)

loading_ratio_calculator = IncrementalMeanCalculator()
construct_representation_calculator = IncrementalMeanCalculator()
forward_propagation_calculator = IncrementalMeanCalculator()

with torch.no_grad():
    model.eval()
    w = args.maximum_load_time / args.max_matrix_len

    # Parameter
    if args.model == "CountMamba":
        time_interval = w * args.time_interval_threshold
    elif args.model == "Holmes":
        open_threshold = 1e-2
        spatial_dist_file = os.path.join(ckp_path, "spatial_distribution.npz")
        spatial_data = np.load(spatial_dist_file)
        webs_centroid = spatial_data["centroid"]
        webs_radius = spatial_data["radius"]

    count = 0
    correct = 0
    loop = tqdm(range(len(test_X)), ncols=150)
    for num in loop:
        data = test_X[num]
        lb = test_y[num]
        count += 1
        # Store
        if args.model == "CountMamba":
            inference_params = InferenceParams()
            window_feature = np.zeros(shape=(6, 2 * (args.maximum_cell_number + 1) + 2))
            history_feature = torch.zeros((1, 1, args.embed_dim)).float().to(device)
            current_index = 0
            last_time = 0.0
        elif args.model == "RF":
            input_DT = []
        elif args.model in ["AWF", "DF", "TMWF"]:
            input_x = []
        elif args.model == "TikTok":
            input_DT = []
        elif args.model == "VarCNN":
            input_d = []
            input_t = []
        elif args.model == "Holmes":
            input_DT = []

        index = -1
        timestamp = np.trim_zeros(data[:, 0], "b")
        while True:
            index += 1
            # receive packets
            window_index = np.where((timestamp < (index + 1) * w) & (timestamp >= index * w))[0]
            receive_data = data[window_index]

            if len(receive_data) != 0:
                # update representation
                start_time = time.time()
                if args.model == "CountMamba":
                    # Generate Feature for this window
                    t_k = receive_data[:, 0]
                    l_k = receive_data[:, 1]
                    d_k = np.sign(l_k).astype(np.int64)
                    c_k = (np.abs(l_k) // 512)
                    c_k = np.clip(c_k, a_max=args.maximum_cell_number, a_min=0.0).astype(np.int64)

                    if index >= 1800:
                        for cell_number in range(args.maximum_cell_number + 1):
                            window_feature[-1, cell_number * 2] += np.sum((d_k < 0) & (c_k == cell_number))
                            window_feature[-1, cell_number * 2 + 1] += np.sum((d_k > 0) & (c_k == cell_number))
                        if window_feature[-1, 2 * args.maximum_cell_number + 2] == 0:
                            window_feature[-1, 2 * args.maximum_cell_number + 2] = 1799 - current_index
                        t_k = np.insert(t_k, 0, last_time)
                        delta_t = np.diff(t_k)
                        cluster_count = np.sum(delta_t > time_interval)
                        window_feature[-1, 2 * args.maximum_cell_number + 3] += cluster_count
                    else:
                        for cell_number in range(args.maximum_cell_number + 1):
                            window_feature[index % 6, cell_number * 2] = np.sum((d_k < 0) & (c_k == cell_number))
                            window_feature[index % 6, cell_number * 2 + 1] = np.sum((d_k > 0) & (c_k == cell_number))
                        window_feature[index % 6, 2 * args.maximum_cell_number + 2] = index - current_index
                        delta_t = np.diff(t_k)
                        cluster_count = np.sum(delta_t > time_interval) + 1
                        window_feature[index % 6, 2 * args.maximum_cell_number + 3] = cluster_count

                    last_time = t_k[-1]
                    current_index = index
                elif args.model == "RF":
                    t_k = receive_data[:, 0]
                    l_k = receive_data[:, 1]
                    d_k = np.sign(l_k).astype(np.int64)

                    sequence = t_k * d_k
                    input_DT.append(sequence)

                    RF_input = np.concatenate(input_DT, axis=0)
                    RF_input = pad_sequence(RF_input, 5000)

                    TAM = process_TAM(RF_input, maximum_load_time=80, max_matrix_len=1800)
                elif args.model in ["AWF", "DF", "TMWF"]:
                    l_k = receive_data[:, 1]
                    d_k = np.sign(l_k).astype(np.int64)
                    input_x.append(d_k)
                elif args.model == "TikTok":
                    t_k = receive_data[:, 0]
                    l_k = receive_data[:, 1]
                    d_k = np.sign(l_k).astype(np.int64)

                    dt_k = d_k * t_k
                    input_DT.append(dt_k)
                elif args.model == "VarCNN":
                    t_k = receive_data[:, 0]
                    l_k = receive_data[:, 1]
                    d_k = np.sign(l_k).astype(np.int64)

                    input_d.append(d_k)
                    input_t.append(t_k)

                elif args.model == "Holmes":
                    t_k = receive_data[:, 0]
                    l_k = receive_data[:, 1]
                    d_k = np.sign(l_k).astype(np.int64)

                    sequence = t_k * d_k
                    sequence *= 1000
                    input_DT.append(sequence)

                    Holmes_input = np.concatenate(input_DT, axis=0)
                    TAF = process_TAF(Holmes_input, interval=40, max_len=2000)

                # end of updating representation
                construct_representation_calculator.add(time.time() - start_time)

            if (index + 1) % 6 == 0:
                start_time = time.time()
                # detection
                if args.model == "CountMamba":
                    x = torch.from_numpy(np.log1p(window_feature).astype(np.float32)).transpose(0, 1)
                    x = x.unsqueeze(0).unsqueeze(0).to(device)
                    result, history_feature = model.forward_step(x, inference_params, history_feature, index // 6,
                                                                 update=index < 1799)
                    result = torch.nn.Softmax(-1)(result)
                    confidence, pred = torch.max(result.flatten(), dim=-1)

                    if index < 1799:
                        window_feature = np.zeros(shape=(6, 2 * (args.maximum_cell_number + 1) + 2))
                elif args.model == "RF":
                    x = torch.from_numpy(TAM.astype(np.float32))
                    x = x.unsqueeze(0).unsqueeze(0).to(device)
                    result = model(x)
                    result = torch.nn.Softmax(-1)(result)
                    confidence, pred = torch.max(result.flatten(), dim=-1)
                elif args.model in ["AWF", "DF", "TMWF"]:
                    if args.model == "AWF":
                        x = pad_sequence(np.concatenate(input_x), 3000)
                    elif args.model == "DF":
                        x = pad_sequence(np.concatenate(input_x), 5000)
                    else:
                        x = pad_sequence(np.concatenate(input_x), 30720)
                    x = torch.from_numpy(x.astype(np.float32))
                    x = x.unsqueeze(0).unsqueeze(0).to(device)
                    result = model(x)
                    result = torch.nn.Softmax(-1)(result)
                    confidence, pred = torch.max(result.flatten(), dim=-1)
                elif args.model == "TikTok":
                    x = pad_sequence(np.concatenate(input_DT), 5000)
                    x = torch.from_numpy(x.astype(np.float32))
                    x = x.unsqueeze(0).unsqueeze(0).to(device)
                    result = model(x)
                    result = torch.nn.Softmax(-1)(result)
                    confidence, pred = torch.max(result.flatten(), dim=-1)
                elif args.model == "VarCNN":
                    xd = pad_sequence(np.concatenate(input_d), 5000)
                    xt = pad_sequence(np.concatenate(input_t), 5000)

                    xt = np.diff(xt)
                    xt[xt < 0] = 0
                    xt = np.insert(xt, 0, 0)

                    x = np.concatenate([xd.reshape(1, -1), xt.reshape(1, -1)], axis=0)
                    x = torch.from_numpy(x.astype(np.float32))
                    x = x.unsqueeze(0).to(device)
                    result = model(x)
                    result = torch.nn.Softmax(-1)(result)
                    confidence, pred = torch.max(result.flatten(), dim=-1)
                elif args.model == "Holmes":
                    x = torch.from_numpy(TAF.astype(np.float32))
                    x = x.unsqueeze(0).to(device)
                    embs = model(x).cpu().numpy()

                    all_sims = 1 - cosine_similarity(embs, webs_centroid)
                    all_sims -= webs_radius

                    confidence, pred = torch.min(torch.from_numpy(all_sims).flatten(), dim=-1)
                    confidence = -1.0 * confidence

                # end of detection
                forward_propagation_calculator.add(time.time() - start_time)

                if confidence.ge(args.threshold):
                    loading_ratio_calculator.add(min((index + 1) * w / np.max(timestamp), 1.0))
                    if pred.item() == lb:
                        correct += 1
                    loop.set_postfix(representation_time=f"{round(construct_representation_calculator.get() * 1000, 2)}ms",
                                     forward_propagation_time=f"{round(forward_propagation_calculator.get() * 1000, 2)}ms")
                    loop.set_description(f"loading_ratio={round(loading_ratio_calculator.get() * 100, 2)}%, "
                                         f"acc={round(correct / count * 100, 2)}%")
                    break

            # final detection
            if w * (index + 1) > np.max(timestamp):
                if (index + 1) % 6 != 0:
                    start_time = time.time()
                    # detection
                    if args.model == "CountMamba":
                        x = torch.from_numpy(np.log1p(window_feature).astype(np.float32)).transpose(0, 1)
                        x = x.unsqueeze(0).unsqueeze(0).to(device)
                        result, history_feature = model.forward_step(x, inference_params, history_feature, index // 6,
                                                                     update=index < 1799)
                        result = torch.nn.Softmax(-1)(result)
                        confidence, pred = torch.max(result.flatten(), dim=-1)
                    elif args.model == "RF":
                        x = torch.from_numpy(TAM.astype(np.float32))
                        x = x.unsqueeze(0).unsqueeze(0).to(device)
                        result = model(x)
                        result = torch.nn.Softmax(-1)(result)
                        confidence, pred = torch.max(result.flatten(), dim=-1)
                    elif args.model in ["AWF", "DF"]:
                        if args.model == "AWF":
                            x = pad_sequence(np.concatenate(input_x), 3000)
                        elif args.model == "DF":
                            x = pad_sequence(np.concatenate(input_x), 5000)
                        else:
                            x = pad_sequence(np.concatenate(input_x), 30720)
                        x = torch.from_numpy(x.astype(np.float32))
                        x = x.unsqueeze(0).unsqueeze(0).to(device)
                        result = model(x)
                        result = torch.nn.Softmax(-1)(result)
                        confidence, pred = torch.max(result.flatten(), dim=-1)
                    elif args.model == "TikTok":
                        x = pad_sequence(np.concatenate(input_DT), 5000)
                        x = torch.from_numpy(x.astype(np.float32))
                        x = x.unsqueeze(0).unsqueeze(0).to(device)
                        result = model(x)
                        result = torch.nn.Softmax(-1)(result)
                        confidence, pred = torch.max(result.flatten(), dim=-1)
                    elif args.model == "VarCNN":
                        xd = pad_sequence(np.concatenate(input_d), 5000)
                        xt = pad_sequence(np.concatenate(input_t), 5000)

                        xt = np.diff(xt)
                        xt[xt < 0] = 0
                        xt = np.insert(xt, 0, 0)

                        x = np.concatenate([xd.reshape(1, -1), xt.reshape(1, -1)], axis=0)
                        x = torch.from_numpy(x.astype(np.float32))
                        x = x.unsqueeze(0).to(device)
                        result = model(x)
                        result = torch.nn.Softmax(-1)(result)
                        confidence, pred = torch.max(result.flatten(), dim=-1)
                    elif args.model == "Holmes":
                        x = torch.from_numpy(TAF.astype(np.float32))
                        x = x.unsqueeze(0).to(device)
                        embs = model(x).cpu().numpy()

                        all_sims = 1 - cosine_similarity(embs, webs_centroid)
                        all_sims -= webs_radius

                        confidence, pred = torch.min(torch.from_numpy(all_sims).flatten(), dim=-1)
                        confidence = -1.0 * confidence

                    # end of detection
                    forward_propagation_calculator.add(time.time() - start_time)

                loading_ratio_calculator.add(min((index + 1) * w / np.max(timestamp), 1.0))
                if pred.item() == lb:
                    correct += 1
                loop.set_postfix(representation_time=f"{round(construct_representation_calculator.get() * 1000, 2)}ms",
                                 forward_propagation_time=f"{round(forward_propagation_calculator.get() * 1000, 2)}ms")
                loop.set_description(f"loading_ratio={round(loading_ratio_calculator.get() * 100, 2)}%, "
                                     f"acc={round(correct / count * 100, 2)}%")
                break

results = {
    "Accuracy": round(100 * correct / count, 2),
    "Loading Ratio": round(100 * loading_ratio_calculator.get(), 2),
    "Construct Representation": round(1000 * construct_representation_calculator.get(), 2),
    "Forward Propagation": round(1000 * forward_propagation_calculator.get(), 2),
}
print(results)

with open(out_file, "w") as fp:
    json.dump(results, fp, indent=4)
