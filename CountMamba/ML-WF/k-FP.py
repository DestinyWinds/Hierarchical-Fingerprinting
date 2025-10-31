import argparse
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


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

ckp_path = os.path.join("../checkpoints", args.dataset, "k-FP")
os.makedirs(ckp_path, exist_ok=True)
out_file = os.path.join(ckp_path, f"max_f1.pth")


# Dataset
def load_data(data_path):
    data = np.load(data_path)
    X = data["X"]
    y = data["y"]

    return X, y


train_X, train_y = load_data(os.path.join("../npz_dataset", args.dataset, f"train.npz"))
num_classes = len(np.unique(train_y))


def extract_feauture(sinste, time_seq):  # sinste: list of packet sizes
    in_sinste = np.abs(sinste[sinste < 0])
    out_sinste = sinste[sinste > 0]

    Totalnumberofpackets = len(sinste)
    Numberofincomingpackets = len(in_sinste)
    Numberofoutgoingpackets = len(out_sinste)

    Numberofincomingpacketsasafraction = Numberofoutgoingpackets / Totalnumberofpackets
    Numberofoutgoingpacketsasafraction = Numberofincomingpackets / Totalnumberofpackets

    MeanOfincomingpacket = np.mean(in_sinste)
    StdOfincomingpacket = np.std(in_sinste)
    MeanOfoutcomingpacket = np.mean(out_sinste)
    StdOfoutcomingpacket = np.std(out_sinste)

    num_chunks = len(sinste) // 20
    chunk_seq = []
    for i in range(num_chunks):
        chunk = sinste[i * 20:(i + 1) * 20]
        chunk_seq.append(np.sum(chunk > 0))

    # chunk_seq =
    std_chunk = np.std(chunk_seq)
    mean_chunk = np.mean(chunk_seq)
    median_chunk = np.median(chunk_seq)
    max_chunk = np.max(chunk_seq)

    ConfirstIn = np.sum(sinste[:30] < 0)
    ConfirstOut = np.sum(sinste[:30] > 0)
    ConlastIn = np.sum(sinste[-30:] < 0)
    ConlastOut = np.sum(sinste[-30:] > 0)

    time_chunk_seq = []
    int_time_seq = time_seq.astype(np.int64)
    for i in range(max(int_time_seq) + 1):
        time_chunk = sinste[int_time_seq == i]
        time_chunk_seq.append(len(time_chunk))

    # time_chunk_seq =
    mean_time_chunk = np.mean(time_chunk_seq)
    std_time_chunk = np.std(time_chunk_seq)
    min_time_chunk = np.min(time_chunk_seq)
    max_time_chunk = np.max(time_chunk_seq)
    median_time_chunk = np.median(time_chunk_seq)

    IAT = np.diff(time_seq)
    IAT = np.insert(IAT, 0, 0)
    in_IAT = np.diff(time_seq[sinste < 0])
    in_IAT = np.insert(in_IAT, 0, 0)
    out_IAT = np.diff(time_seq[sinste > 0])
    out_IAT = np.insert(out_IAT, 0, 0)

    max_IAT = np.max(IAT)
    mean_IAT = np.mean(IAT)
    std_IAT = np.std(IAT)
    qua_IAT = np.percentile(IAT, 75)

    max_in_IAT = np.max(in_IAT)
    mean_in_IAT = np.mean(in_IAT)
    std_in_IAT = np.std(in_IAT)
    qua_in_IAT = np.percentile(in_IAT, 75)

    max_out_IAT = np.max(out_IAT)
    mean_out_IAT = np.mean(out_IAT)
    std_out_IAT = np.std(out_IAT)
    qua_out_IAT = np.percentile(out_IAT, 75)

    in_time_seq = time_seq[sinste < 0]
    out_time_seq = time_seq[sinste > 0]

    first_time_quart = np.percentile(time_seq, 0.25)
    second_time_quart = np.percentile(time_seq, 0.5)
    third_time_quart = np.percentile(time_seq, 0.75)
    fourth_time_quart = np.max(time_seq)

    if len(in_time_seq) > 0:
        in_first_time_quart = np.percentile(in_time_seq, 0.25)
        in_second_time_quart = np.percentile(in_time_seq, 0.5)
        in_third_time_quart = np.percentile(in_time_seq, 0.75)
        in_fourth_time_quart = np.max(in_time_seq)
    else:
        in_first_time_quart = 0
        in_second_time_quart = 0
        in_third_time_quart = 0
        in_fourth_time_quart = 0

    if len(out_time_seq) > 0:
        out_first_time_quart = np.percentile(out_time_seq, 0.25)
        out_second_time_quart = np.percentile(out_time_seq, 0.5)
        out_third_time_quart = np.percentile(out_time_seq, 0.75)
        out_fourth_time_quart = np.max(out_time_seq)
    else:
        out_first_time_quart = 0
        out_second_time_quart = 0
        out_third_time_quart = 0
        out_fourth_time_quart = 0

    subset_size = len(chunk_seq) // 20
    Alternativecon = [np.sum(chunk_seq[i * subset_size: (i + 1) * subset_size]) for i in range(20)]

    subset_size = len(time_chunk_seq) // 20
    Alternativecon2 = [np.sum(time_chunk_seq[i * subset_size: (i + 1) * subset_size]) for i in range(20)]

    features = [Totalnumberofpackets, Numberofincomingpackets, Numberofoutgoingpackets,
                Numberofincomingpacketsasafraction, Numberofoutgoingpacketsasafraction,
                MeanOfincomingpacket, StdOfincomingpacket, MeanOfoutcomingpacket, StdOfoutcomingpacket,
                std_chunk, mean_chunk, median_chunk, max_chunk,
                ConfirstIn, ConfirstOut, ConlastIn, ConlastOut,
                mean_time_chunk, std_time_chunk, min_time_chunk, max_time_chunk, median_time_chunk,
                max_IAT, mean_IAT, std_IAT, qua_IAT,
                max_in_IAT, mean_in_IAT, std_in_IAT, qua_in_IAT,
                max_out_IAT, mean_out_IAT, std_out_IAT, qua_out_IAT,
                first_time_quart, second_time_quart, third_time_quart, fourth_time_quart,
                in_first_time_quart, in_second_time_quart, in_third_time_quart, in_fourth_time_quart,
                out_first_time_quart, out_second_time_quart, out_third_time_quart, out_fourth_time_quart]
    features.extend(Alternativecon)
    features.extend(Alternativecon2)
    return features


def extract(X):
    features = []
    for data in tqdm(X):
        packet_time = data[:, 0]
        packet_length = data[:, 1]
        feature = extract_feauture(packet_length, packet_time)
        features.append(feature)
    features = np.array(features)
    features = np.nan_to_num(features, nan=0)
    return features


train_features = extract(train_X)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(train_features, train_y)
joblib.dump(rf, out_file)
