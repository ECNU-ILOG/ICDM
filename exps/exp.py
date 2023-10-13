import os
from pprint import pprint
import math
import wandb as wb
import random
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

current_path = os.path.abspath('.')
tmp = os.path.dirname(current_path)
path_CDM_ILOG = os.path.dirname(tmp)
path_CDM_ICDM_runner = path_CDM_ILOG + '\\runners\\ICDM'
sys.path.insert(0, tmp)
sys.path.insert(0, path_CDM_ILOG)
sys.path.insert(0, path_CDM_ICDM_runner)

from data.data_params_dict import data_params
from runners.ICDM.utils import epochs_dict, build_graph4CE, build_graph4SE, build_graph4SC
from runners.ICDM.cdm_runners import get_runner
from runners.commonutils.util import set_seeds
from runners.ICDM.utils import save

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--method', default='icdm', type=str,
                    help='A Lightweight Graph-based Cognitive Diagnosis Framework', required=True)
parser.add_argument('--datatype', default='junyi', type=str, help='benchmark', required=True)
parser.add_argument('--test_size', default=0.2, type=float, help='test size of benchmark', required=True)
parser.add_argument('--epoch', type=int, help='epoch of method')
parser.add_argument('--seed', default=0, type=int, help='seed for exp', required=True)
parser.add_argument('--dtype', default=torch.float64, help='dtype of tensor')
parser.add_argument('--device', default='cuda', type=str, help='device for exp')
parser.add_argument('--gcnlayers', type=int, help='numbers of gcn layers')
parser.add_argument('--dim', type=int, help='dimension of hidden layer')
parser.add_argument('--batch_size', type=int, help='batch size of benchmark')
parser.add_argument('--exp_type', help='experiment type')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--agg_type', type=str, help='the type of aggregator')
parser.add_argument('--cdm_type', type=str, help='the inherent CDM', default='lightgcn')
parser.add_argument('--khop', type=int)
config_dict = vars(parser.parse_args())

name = f"{config_dict['method']}-{config_dict['cdm_type']}-{config_dict['datatype']}-seed{config_dict['seed']}"
if 'icdm' in config_dict['method']:
    name = f"{config_dict['method']}-{config_dict['cdm_type']}-{config_dict['datatype']}-seed{config_dict['seed']}"
    config_dict['method'] = f"{config_dict['method']}-{config_dict['cdm_type']}"
else:
    name = f"{config_dict['method']}-{config_dict['datatype']}-seed{config_dict['seed']}"
tags = [config_dict['method'], config_dict['datatype'], str(config_dict['seed'])]
config_dict['name'] = name
method = config_dict['method']
datatype = config_dict['datatype']

if config_dict.get('epoch', None) is None:
    config_dict['epoch'] = epochs_dict[datatype][method]

if config_dict.get('batch_size', None) is None:
    config_dict['batch_size'] = data_params[datatype]['batch_size']
if 'icdm' in method:
    if config_dict.get('weight_reg') is None:
        config_dict['weight_reg'] = 1e-3
pprint(config_dict)
run = wb.init(project="ICDM", name=name,
              tags=tags,
              config=config_dict)
config_dict['id'] = run.id


def main(config):
    method = config['method']
    runner = get_runner(method)
    datatype = config['datatype']
    device = config['device']
    dtype = config['dtype']
    torch.set_default_dtype(dtype)
    config.update({
        'stu_num': data_params[datatype]['stu_num'],
        'prob_num': data_params[datatype]['prob_num'],
        'know_num': data_params[datatype]['know_num'],
    })
    set_seeds(config['seed'])
    q_np = pd.read_csv('../data/{}/q.csv'.format(datatype),
                       header=None).to_numpy()
    q_tensor = torch.tensor(q_np).to(device)
    exp_type = config['exp_type']
    if not os.path.exists(f'logs/{exp_type}'):
        os.makedirs(f'logs/{exp_type}')
    if exp_type == 'concept':
        np_train = pd.read_csv('../data/{}/{}TrainData.csv'.format(datatype, datatype),
                               header=None).to_numpy()
        np_test = pd.read_csv('../data/{}/{}TestData.csv'.format(datatype, datatype),
                              header=None).to_numpy()
        directed_graph = pd.read_csv('../data/{}/directed_graph.csv'.format(datatype),
                                     header=None).to_numpy()
        config['directed_graph'] = directed_graph
        undirected_graph = pd.read_csv('../data/{}/undirected_graph.csv'.format(datatype),
                                       header=None).to_numpy()
        config['undirected_graph'] = undirected_graph
    elif exp_type == 'bad':
        np_data = pd.read_csv('../data/{}/{}TotalData-{}.csv'.format(datatype, datatype, config['bad_ratio']),
                              header=None).to_numpy()
        np_train, np_test = train_test_split(np_data, test_size=config['test_size'], random_state=config['seed'])
    else:
        np_data = pd.read_csv('../data/{}/{}TotalData.csv'.format(datatype, datatype),
                              header=None).to_numpy()
        np_train, np_test = train_test_split(np_data, test_size=config['test_size'], random_state=config['seed'])
    if exp_type == 'noise':
        noise_ratio = config['noise_ratio']
        train_size = np_train.shape[0]
        noise_index = np.random.choice(np.arange(train_size), size=int(noise_ratio * train_size))
        for index in noise_index:
            init_score = np_train[index, 2]
            if init_score == 1:
                np_train[index, 2] = 0
            else:
                np_train[index, 2] = 1
    config['np_train'] = np_train
    config['np_test'] = np_test
    config['q'] = q_tensor
    # C = get_correlate_matrix(datatype, np_train)
    # config['C'] = torch.tensor(C).to(device)
    right, wrong = build_graph4SE(config)
    graph_dict = {
        'right': right,
        'wrong': wrong,
        'Q': build_graph4CE(config),
        'I': build_graph4SC(config)
    }
    config['graph_dict'] = graph_dict
    runner(config, save)


if __name__ == '__main__':
    sys.exit(main(config_dict))
