import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from util import process_TAM, pad_sequence


class RFDataset(Dataset):
    def __init__(self, X, Y, length, loaded_ratio=100):
        self.X = X
        self.Y = Y
        self.length = length
        self.loaded_ratio = loaded_ratio

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        data = self.X[index]
        label = self.Y[index]

        timestamp = data[:, 0]
        sign = np.sign(data[:, 1])
        X = timestamp * sign

        loading_time = timestamp.max()
        threshold = loading_time * self.loaded_ratio / 100

        timestamp = np.trim_zeros(timestamp, "b")
        valid_index = np.where(timestamp <= threshold)

        X = X[valid_index]

        X = pad_sequence(X, self.length)

        return self.process_data(X), label

    def process_data(self, data):
        TAM = process_TAM(data, maximum_load_time=80, max_matrix_len=1800)
        TAM = TAM.reshape(1, 2, 1800)

        return TAM.astype(np.float32)


class DT2Dataset(Dataset):
    def __init__(self, X, Y, length, loaded_ratio=100):
        self.X = X
        self.Y = Y
        self.length = length
        self.loaded_ratio = loaded_ratio

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        data = self.X[index]
        label = self.Y[index]

        timestamp = data[:, 0]
        sign = np.sign(data[:, 1])

        loading_time = timestamp.max()
        threshold = loading_time * self.loaded_ratio / 100

        timestamp = np.trim_zeros(timestamp, "b")
        valid_index = np.where(timestamp <= threshold)

        timestamp = timestamp[valid_index]
        sign = sign[valid_index]

        timestamp = pad_sequence(timestamp, self.length)
        sign = pad_sequence(sign, self.length)

        return self.process_data(timestamp, sign), label

    def process_data(self, X_time, X_dir):
        X_time = np.diff(X_time)
        X_time[X_time < 0] = 0
        X_time = np.insert(X_time, 0, 0)

        X_dir = X_dir.reshape(1, -1)
        X_time = X_time.reshape(1, -1)
        data = np.concatenate([X_dir, X_time], axis=0)

        return data.astype(np.float32)


class DTDataset(Dataset):
    def __init__(self, X, Y, length, loaded_ratio=100):
        self.X = X
        self.Y = Y
        self.length = length
        self.loaded_ratio = loaded_ratio

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        data = self.X[index]
        label = self.Y[index]

        timestamp = data[:, 0]
        sign = np.sign(data[:, 1])
        dt = timestamp * sign

        loading_time = timestamp.max()
        threshold = loading_time * self.loaded_ratio / 100

        timestamp = np.trim_zeros(timestamp, "b")
        valid_index = np.where(timestamp <= threshold)

        dt = dt[valid_index]

        return self.process_data(dt), label

    def process_data(self, dt):
        dt = pad_sequence(dt, self.length)

        return dt.reshape(1, -1).astype(np.float32)


class DirectionDataset(Dataset):
    def __init__(self, X, Y, length, loaded_ratio=100):
        self.X = X
        self.Y = Y
        self.length = length
        self.loaded_ratio = loaded_ratio

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        data = self.X[index]
        label = self.Y[index]

        timestamp = data[:, 0]
        sign = np.sign(data[:, 1])

        loading_time = timestamp.max()
        threshold = loading_time * self.loaded_ratio / 100

        timestamp = np.trim_zeros(timestamp, "b")
        valid_index = np.where(timestamp <= threshold)

        sign = sign[valid_index]

        return self.process_data(sign), label

    def process_data(self, direction):
        direction = pad_sequence(direction, self.length)

        return direction.reshape(1, -1).astype(np.float32)
