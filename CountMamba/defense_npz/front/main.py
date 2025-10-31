import numpy as np
import argparse
import logging
import sys
import pandas as pd
import os
from os.path import join
from os import makedirs
import constants as ct
from time import strftime
import matplotlib.pyplot as plt
import multiprocessing as mp
import configparser
import time
import datetime
from pprint import pprint
import glob
import tqdm

logger = logging.getLogger('ranpad2')


def init_directories():
    # Create a results dir if it doesn't exist yet
    if not os.path.exists(ct.RESULTS_DIR):
        makedirs(ct.RESULTS_DIR)

    # Define output directory
    timestamp = strftime('%m%d_%H%M')
    output_dir = join(ct.RESULTS_DIR, 'ranpad2_' + timestamp)
    makedirs(output_dir)

    return output_dir


def config_logger(args):
    # Set file
    log_file = sys.stdout
    if args.log != 'stdout':
        log_file = open(args.log, 'w')
    ch = logging.StreamHandler(log_file)

    # Set logging format
    ch.setFormatter(logging.Formatter(ct.LOG_FORMAT))
    logger.addHandler(ch)

    # Set level format
    logger.setLevel(logging.INFO)


def parse_arguments():
    conf_parser = configparser.RawConfigParser()
    conf_parser.read(ct.CONFIG_FILE)

    parser = argparse.ArgumentParser(description='It simulates adaptive padding on a set of web traffic traces.')

    parser.add_argument('--p',
                        metavar='<traces path>',
                        default="../../npz_dataset/Closed_2tab",
                        help='Path to the directory with the traffic traces to be simulated.')
    parser.add_argument('-format',
                        metavar='<suffix of a file>',
                        default='',
                        help='suffix of a file.')
    parser.add_argument('-c', '--config',
                        dest="section",
                        metavar='<config name>',
                        help="Adaptive padding configuration.",
                        choices=conf_parser.sections(),
                        default="default")

    parser.add_argument('--log',
                        type=str,
                        dest="log",
                        metavar='<log path>',
                        default='stdout',
                        help='path to the log file. It will print to stdout by default.')

    args = parser.parse_args()
    config = dict(conf_parser._sections[args.section])
    config_logger(args)
    return args, config


def load_trace(fdir):
    with open(fdir, 'r') as f:
        tmp = f.readlines()

    t = pd.Series(tmp).str.slice(0, -1).str.split('\t', expand=True).astype('float')
    return np.array(t)


def dump(trace, fname):
    global output_dir
    with open(join(output_dir, fname), 'w') as fo:
        for packet in trace:
            fo.write("{:.4f}".format(packet[0]) + '\t' + "{}".format(int(packet[1])) \
                     + ct.NL)


def simulate(data):
    while data.shape[0] > 0 and np.all(data[-1] == 0):
        data = data[:-1]

    np.random.seed(datetime.datetime.now().microsecond)
    ori_trace = np.array(data)
    trace = RP(ori_trace)

    return ori_trace, trace


def RP(trace):
    # format: [[time, pkt],[...]]
    # trace, cpkt_num, spkt_num, cwnd, swnd
    global client_dummy_pkt_num
    global server_dummy_pkt_num
    global min_wnd
    global max_wnd
    global start_padding_time
    global client_min_dummy_pkt_num
    global server_min_dummy_pkt_num

    client_wnd = np.random.uniform(min_wnd, max_wnd)
    server_wnd = np.random.uniform(min_wnd, max_wnd)
    if client_min_dummy_pkt_num != client_dummy_pkt_num:
        client_dummy_pkt = np.random.randint(client_min_dummy_pkt_num, client_dummy_pkt_num)
    else:
        client_dummy_pkt = client_dummy_pkt_num
    if server_min_dummy_pkt_num != server_dummy_pkt_num:
        server_dummy_pkt = np.random.randint(server_min_dummy_pkt_num, server_dummy_pkt_num)
    else:
        server_dummy_pkt = server_dummy_pkt_num
    logger.debug("client_wnd:", client_wnd)
    logger.debug("server_wnd:", server_wnd)
    logger.debug("client pkt:", client_dummy_pkt)
    logger.debug("server pkt:", server_dummy_pkt)

    first_incoming_pkt_time = trace[np.where(trace[:, 1] < 0)][0][0]
    last_pkt_time = trace[-1][0]

    client_timetable = getTimestamps(client_wnd, client_dummy_pkt)
    client_timetable = client_timetable[np.where(start_padding_time + client_timetable[:, 0] <= last_pkt_time)]

    server_timetable = getTimestamps(server_wnd, server_dummy_pkt)
    server_timetable[:, 0] += first_incoming_pkt_time
    server_timetable = server_timetable[np.where(start_padding_time + server_timetable[:, 0] <= last_pkt_time)]

    # print("client_timetable")
    # print(client_timetable[:10])
    client_pkts = np.concatenate((client_timetable, 1 * np.ones((len(client_timetable), 1))), axis=1)
    server_pkts = np.concatenate((server_timetable, -1 * np.ones((len(server_timetable), 1))), axis=1)

    noisy_trace = np.concatenate((trace, client_pkts, server_pkts), axis=0)
    noisy_trace = noisy_trace[noisy_trace[:, 0].argsort(kind='mergesort')]
    return noisy_trace


def getTimestamps(wnd, num):
    # timestamps = sorted(np.random.exponential(wnd/2.0, num))   
    # print(wnd, num)
    # timestamps = sorted(abs(np.random.normal(0, wnd, num)))
    timestamps = sorted(np.random.rayleigh(wnd, num))
    # print(timestamps[:5])
    # timestamps = np.fromiter(map(lambda x: x if x <= wnd else wnd, timestamps),dtype = float)
    return np.reshape(timestamps, (len(timestamps), 1))


def parallel(flist, n_jobs=20):
    pool = mp.Pool(n_jobs)
    pool.map(simulate, flist)


if __name__ == '__main__':
    global client_dummy_pkt_num
    global server_dummy_pkt_num
    global client_min_dummy_pkt_num
    global server_min_dummy_pkt_num
    global max_wnd
    global min_wnd
    global start_padding_time
    args, config = parse_arguments()
    logger.info(args)

    client_min_dummy_pkt_num = int(config.get('client_min_dummy_pkt_num', 1))
    server_min_dummy_pkt_num = int(config.get('server_min_dummy_pkt_num', 1))
    client_dummy_pkt_num = int(config.get('client_dummy_pkt_num', 300))
    server_dummy_pkt_num = int(config.get('server_dummy_pkt_num', 300))
    start_padding_time = int(config.get('start_padding_time', 0))
    max_wnd = float(config.get('max_wnd', 10))
    min_wnd = float(config.get('min_wnd', 10))

    print("client_min_dummy_pkt_num:{}".format(client_min_dummy_pkt_num))
    print("server_min_dummy_pkt_num:{}".format(server_min_dummy_pkt_num))
    print("client_dummy_pkt_num: {}\nserver_dummy_pkt_num: {}".format(client_dummy_pkt_num, server_dummy_pkt_num))
    print("max_wnd: {}\nmin_wnd: {}".format(max_wnd, min_wnd))
    print("start_padding_time:", start_padding_time)

    data = np.load(args.p + "/data.npz")
    X = data["X"]
    y = data["y"]

    # Init run directories
    output_dir = init_directories()
    logger.info("Traces are dumped to {}".format(output_dir))
    start = time.time()
    # for i,f in enumerate(flist):
    #     logger.debug('Simulating {}'.format(f))
    #     if i %2000 == 0:
    #         print(r"Done for inst ",i,flush = True)
    #     simulate(f)

    # for filename in tqdm.tqdm(flist):
    #     simulate(filename)

    def latency(trace):
        if len(trace) < 2:
            return 0.0
        return trace[-1, 0] - trace[0, 0]

    def bandwidth(trace):
        total_bytes = np.sum(np.abs(trace[:, 1]))
        return 1.0 * total_bytes / latency(trace)

    def bandwidth_ovhd(new, old):
        bw_old = bandwidth(old)
        if bw_old == 0.0:
            return 0.0
        return 1.0 * bandwidth(new) / bw_old

    def latency_ovhd(new, old):
        lat_old = latency(old)
        if lat_old == 0.0:
            return 0.0
        return 1.0 * latency(new) / lat_old

    latencies, bandwidths = [], []
    key_array = np.lib.format.open_memmap("large_data_key.npy", mode="w+", dtype="float32", shape=X.shape)
    count = -1
    for p, label in tqdm.tqdm(zip(X, y)):
        count += 1
        trace, simulated = simulate(p)

        for i in range(len(simulated)):
            key_array[count, i, 0] = simulated[i, 0]
            key_array[count, i, 1] = simulated[i, 1]

            if i == key_array.shape[1] - 1:
                break

        bw_ovhd = bandwidth_ovhd(simulated, trace)
        bandwidths.append(bw_ovhd)
        logger.debug("Bandwidth overhead: %s" % bw_ovhd)

        lat_ovhd = latency_ovhd(simulated, trace)
        latencies.append(lat_ovhd)

    np.savez(os.path.join(output_dir, "data.npz"), X=key_array, y=y)
    os.remove("large_data_key.npy")

    logger.info("Latency overhead: %s" % np.median([l for l in latencies if l > 0.0]))
    logger.info("Bandwidth overhead: %s" % np.median([b for b in bandwidths if b > 0.0]))
    
    # parallel(flist)
    logger.info("Time: {}".format(time.time() - start))
