from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import math


class CountDataset(Dataset):
    def __init__(self, X, labels, args, BAPM=None):
        self.X = X
        self.labels = labels
        self.args = args
        self.BAPM = BAPM

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        data = self.X[index]
        label = self.labels[index]

        if self.BAPM is not None:
            bapm = self.BAPM[index]
            return self.process_data(data, bapm=bapm), label
        else:
            return self.process_data(data), label

    def process_data(self, data, bapm=None):
        time = data[:, 0]
        packet_length = data[:, 1]

        packet_length = pad_sequence(packet_length, self.args.seq_len)
        time = pad_sequence(time, self.args.seq_len)

        TAM, current_index, bapm_labels = process_CountMatrix(packet_length, time, args=self.args, bapm=bapm)

        TAM = TAM.reshape((1, 2 * (self.args.maximum_cell_number + 1) + 2, self.args.max_matrix_len))

        if self.args.log_transform:
            TAM = np.log1p(TAM)

        if bapm is not None:
            return TAM.astype(np.float32), current_index, bapm_labels
        else:
            return TAM.astype(np.float32), current_index


def pad_sequence(sequence, length):
    if len(sequence) >= length:
        sequence = sequence[:length]
    else:
        sequence = np.pad(sequence, (0, length - len(sequence)), "constant", constant_values=0.0)

    return sequence


def process_CountMatrix(packet_length, time, args, bapm):
    feature = np.zeros((2 * (args.maximum_cell_number + 2), args.max_matrix_len))
    w = args.maximum_load_time / args.max_matrix_len
    time_interval = w * args.time_interval_threshold

    if bapm is not None:
        bapm_time = np.trim_zeros(time, "b")

        bapm_first_start = min(len(bapm_time) - 1, bapm[0])
        bapm_first_end = min(len(bapm_time) - 1, bapm[0] + bapm[1])
        bapm_second_start = min(len(bapm_time) - 1, bapm[2])
        bapm_second_end = min(len(bapm_time) - 1, bapm[2] + bapm[3])

        bapm_first_start_time = bapm_time[bapm_first_start]
        bapm_first_end_time = bapm_time[bapm_first_end]
        bapm_second_start_time = bapm_time[bapm_second_start]
        bapm_second_end_time = bapm_time[bapm_second_end]

        bapm_first_start_position = min(math.floor(bapm_first_start_time / w), args.max_matrix_len - 1)
        bapm_first_end_position = min(math.floor(bapm_first_end_time / w), args.max_matrix_len - 1)
        bapm_second_start_position = min(math.floor(bapm_second_start_time / w), args.max_matrix_len - 1)
        bapm_second_end_position = min(math.floor(bapm_second_end_time / w), args.max_matrix_len - 1)

        bapm_labels = np.full((2, args.max_matrix_len), -1)
        bapm_labels[0, bapm_first_start_position:bapm_first_end_position + 1] = bapm[-2]
        bapm_labels[1, bapm_second_start_position:bapm_second_end_position + 1] = bapm[-1]
    else:
        bapm_labels = None

    current_index = 0
    current_timestamps = []
    for l_k, t_k in zip(packet_length, time):
        if t_k == 0 and l_k == 0:
            break  # End of sequence

        d_k = int(np.sign(l_k))
        c_k = min(int(np.abs(l_k) // 512), args.maximum_cell_number)  # [0, C]

        fragment = 0 if d_k < 0 else 1
        i = 2 * c_k + fragment  # [0, 2C + 1]
        j = min(math.floor(t_k / w), args.max_matrix_len - 1)
        j = max(j, 0)
        feature[i, j] += 1

        if j != current_index:
            feature[2 * args.maximum_cell_number + 2, j] = max(j - current_index, 0)

            delta_t = np.diff(current_timestamps)
            cluster_count = np.sum(delta_t > time_interval) + 1
            feature[2 * args.maximum_cell_number + 3, current_index] = cluster_count

            current_index = j
            current_timestamps = [t_k]
        else:
            current_timestamps.append(t_k)

    delta_t = np.diff(current_timestamps)
    cluster_count = np.sum(delta_t > time_interval) + 1
    feature[2 * args.maximum_cell_number + 3, current_index] = cluster_count

    return feature, current_index, bapm_labels

