import os
from pprint import pprint
import math
import wandb as wb
import random
import torch
import numpy as np
import pandas as pd
import sys
import pickle

current_path = os.path.abspath('.')
tmp = os.path.dirname(current_path)
path_CDM_ILOG = os.path.dirname(tmp)
path_CDM_ICDM_runner = path_CDM_ILOG + '\\runners\\ICDM'
sys.path.insert(0, tmp)
sys.path.insert(0, path_CDM_ILOG)
sys.path.insert(0, path_CDM_ICDM_runner)
from runners.ICDM.utils import epochs_dict, build_graph4CE, build_graph4SE, build_graph4SC
from runners.ICDM.cdm_runners import get_ind_runner
from data.data_params_dict import data_params
from runners.commonutils.util import set_seeds
from runners.commonutils.datautils import get_split_by_student, get_split_by_student_acr, get_split_inductive
import argparse
from runners.ICDM.utils import save

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
parser.add_argument('--new_ratio', type=float, help='the ratio of new students')
parser.add_argument('--known_dis', type=str, help='the known distribution of students')
parser.add_argument('--mode', type=str, help='method mode')
parser.add_argument('--cdm_type', type=str, help='the inherent interaction type', default='lightgcn')
parser.add_argument('--khop', type=int, help='the k-hop neighbor of certain nodes')
parser.add_argument('--ab', type=str, help='the ablation study of igcdm')
parser.add_argument('--weight_reg', type=float, help='the ablation study of igcdm')
parser.add_argument('--d_1', type=float, default=0.1)
parser.add_argument('--d_2', type=float, default=0.2)
config_dict = vars(parser.parse_args())

if 'icdm' in config_dict['method'] or 'imcgae' in config_dict['method']:
    name = f"{config_dict['method']}-{config_dict['cdm_type']}-{config_dict['datatype']}-seed{config_dict['seed']}"
    config_dict['method'] = f"{config_dict['method']}-{config_dict['cdm_type']}"
    tags = [config_dict['method'], config_dict['datatype'], str(config_dict['seed'])]
    config_dict['name'] = name
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
    runner = get_ind_runner(method)
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
    if exp_type == 'ind' or exp_type == 'sparse' or exp_type == 'gcn' or exp_type == 'dim' or exp_type == 'drop' or exp_type == 'ab' or exp_type == 'khop' or exp_type=='agg' or exp_type=='reg':
        np_train_old, np_train_new, np_test_all, np_test_new, exist_idx, new_idx = get_split_inductive(datatype,
                                                                                                       test_size=config[
                                                                                                           'test_size'],
                                                                                                       new_student_ratio=
                                                                                                       config[
                                                                                                           'new_ratio'],
                                                                                                       seed=config[
                                                                                                           'seed'])
    elif exp_type == 'dis':
        np_train, np_test, exist_idx, new_idx = get_split_by_student_acr(datatype, seed=config['seed'],
                                                                         new_student_ratio=config['new_ratio'],
                                                                         known_dis=config['known_dis'])
    else:
        raise ValueError('we do not have this exp_type')

    config['np_train_old'] = np_train_old
    config['np_train_new'] = np_train_new
    config['np_test'] = np_test_all
    config['np_test_new'] = np_test_new
    config['q'] = q_tensor
    config['exist_idx'] = exist_idx
    config['new_idx'] = new_idx
    if 'icdm-re' not in method:
        right_old, wrong_old = build_graph4SE(config, mode='ind_train')
        right_eval, wrong_eval = build_graph4SE(config, mode='ind_eval')
        graph_dict = {
            'right_old': right_old,
            'wrong_old': wrong_old,
            'right_eval': right_eval,
            'wrong_eval': wrong_eval,
            'Q': build_graph4CE(config),
            'I_old': build_graph4SC(config, mode='ind_train'),
            'I_eval': build_graph4SC(config, mode='ind_eval'),
            'involve': build_graph4SC(config, mode='involve')
        }
        config['graph_dict'] = graph_dict
    else:
        config['np_train'] = np.vstack((config['np_train_old'], config['np_train_new']))
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
