import glob
from tqdm import tqdm
import os


def get_trace(file_name):
    all_lines = []
    with open(file_name) as fptr:
        for line in fptr:
            time = float(line.strip().split('\t')[0])
            packet_length = int(line.strip().split('\t')[1])

            trace_line = (time, packet_length)
            all_lines.append(trace_line)
    return all_lines


def merge_tuples(tuples):
    merged = []
    current_key, current_sum = tuples[0]

    for key, value in tuples[1:]:
        if key == current_key:
            current_sum += value
        else:
            merged.append((current_key, current_sum))
            current_key, current_sum = key, value

    merged.append((current_key, current_sum))

    return merged


filenames = glob.glob("../dataset/k-NN/*") + glob.glob("../dataset/W_T/*")
filenames = [filename for filename in filenames if "format" not in filename.split("-")[-1]]
for filename in tqdm(filenames):
    trace = get_trace(filename)
    formatted_trace = merge_tuples(trace)
    formatted_trace = [(key, value * 512) for key, value in formatted_trace]

    new_filename = filename + "format"
    with open(new_filename, 'w') as w:
        for p in formatted_trace:
            w.write(str(p[0]) + '\t' + str(p[1]) + '\n')


delete_filenames = glob.glob("../dataset/k-NN/*") + glob.glob("../dataset/W_T/*")
delete_filenames = [filename for filename in delete_filenames if "format" not in filename.split("-")[-1]]
for filename in tqdm(delete_filenames):
    os.remove(filename)
