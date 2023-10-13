import dgl
import torch
import numpy as np
import pandas as pd
import networkx as nx
from data.data_params_dict import data_params


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


def l2_loss(*weights):
    loss = 0.0
    for w in weights:
        loss += torch.sum(torch.pow(w, 2)) / w.shape[0]

    return 0.5 * loss


def build_graph4CE(config: dict):
    q = config['q']
    q = q.detach().cpu().numpy()
    know_num = config['know_num']
    exer_num = config['prob_num']
    node = exer_num + know_num
    g = dgl.DGLGraph()
    g.add_nodes(node)
    edge_list = []
    indices = np.where(q != 0)
    for exer_id, know_id in zip(indices[0].tolist(), indices[1].tolist()):
        edge_list.append((int(know_id + exer_num), int(exer_id)))
        edge_list.append((int(exer_id), int(know_id + exer_num)))
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    return g


def build_graph4SE(config: dict, mode='tl'):
    if mode == 'tl':
        data = config['np_train']
    elif mode == 'ind_train':
        data = config['np_train_old']
    else:
        data = np.vstack((config['np_train_old'], config['np_train_new']))

    stu_num = config['stu_num']
    exer_num = config['prob_num']
    node = stu_num + exer_num
    g_right = dgl.DGLGraph()
    g_right.add_nodes(node)
    g_wrong = dgl.DGLGraph()
    g_wrong.add_nodes(node)

    right_edge_list = []
    wrong_edge_list = []
    for index in range(data.shape[0]):
        stu_id = data[index, 0]
        exer_id = data[index, 1]
        if int(data[index, 2]) == 1:
            if mode == 'tl' or mode == 'ind_train' or int(stu_id) in config['exist_idx']:
                right_edge_list.append((int(stu_id), int(exer_id + stu_num)))
                right_edge_list.append((int(exer_id + stu_num), int(stu_id)))
            else:
                right_edge_list.append((int(exer_id + stu_num), int(stu_id)))
        else:
            if mode == 'tl' or mode == 'ind_train' or int(stu_id) in config['exist_idx']:
                wrong_edge_list.append((int(stu_id), int(exer_id + stu_num)))
                wrong_edge_list.append((int(exer_id + stu_num), int(stu_id)))
            else:
                wrong_edge_list.append((int(exer_id + stu_num), int(stu_id)))
    right_src, right_dst = tuple(zip(*right_edge_list))
    wrong_src, wrong_dst = tuple(zip(*wrong_edge_list))
    g_right.add_edges(right_src, right_dst)
    g_wrong.add_edges(wrong_src, wrong_dst)
    return g_right, g_wrong


def build_graph4SC(config: dict, mode='tl'):
    if mode == 'tl':
        data = config['np_train']
    elif mode == 'ind_train':
        data = config['np_train_old']
    else:
        data = np.vstack((config['np_train_old'], config['np_train_new']))
    stu_num = config['stu_num']
    know_num = config['know_num']
    q = config['q']
    q = q.detach().cpu().numpy()
    node = stu_num + know_num
    g = dgl.DGLGraph()
    g.add_nodes(node)
    edge_list = []
    sc_matrix = np.zeros(shape=(stu_num, know_num))
    for index in range(data.shape[0]):
        stu_id = data[index, 0]
        exer_id = data[index, 1]
        concepts = np.where(q[int(exer_id)] != 0)[0]
        for concept_id in concepts:
            if mode == 'tl' or mode == 'ind_train' or int(stu_id) in config['exist_idx']:
                if sc_matrix[int(stu_id), int(concept_id)] != 1:
                    edge_list.append((int(stu_id), int(concept_id + stu_num)))
                    edge_list.append((int(concept_id + stu_num), int(stu_id)))
                    sc_matrix[int(stu_id), int(concept_id)] = 1
            else:
                if mode != 'involve':
                    if sc_matrix[int(stu_id), int(concept_id)] != 1:
                        edge_list.append((int(concept_id + stu_num), int(stu_id)))
                        sc_matrix[int(stu_id), int(concept_id)] = 1
                else:
                    if sc_matrix[int(stu_id), int(concept_id)] != 1:
                        edge_list.append((int(stu_id), int(concept_id + stu_num)))
                        sc_matrix[int(stu_id), int(concept_id)] = 1
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    return g


def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity


def softmax(vector):
    exp_vector = np.exp(vector)
    sum_exp = np.sum(exp_vector)
    softmax_vector = exp_vector / sum_exp
    return softmax_vector


def get_train_R_matrix(datatype, data):
    stu_num = data_params[datatype]['stu_num']
    prob_num = data_params[datatype]['prob_num']
    R_matrix = np.zeros(shape=(stu_num, prob_num))
    for k in range(data.shape[0]):
        if data[k, 2] == 1:
            R_matrix[int(data[k, 0]), int(data[k, 1])] = 1
        else:
            R_matrix[int(data[k, 0]), int(data[k, 1])] = -1
    return R_matrix


def dgl2tensor(g):
    nx_graph = g.to_networkx()
    adj_matrix = nx.to_numpy_matrix(nx_graph)
    tensor = torch.from_numpy(adj_matrix)
    return tensor


def concept_distill(matrix, concept):
    coeff = 1.0 / torch.sum(matrix, dim=1)
    concept = matrix.to(torch.float64) @ concept
    concept_distill = concept * coeff[:, None]
    return concept_distill


def get_subgraph(g, id, device):
    return dgl.in_subgraph(g, id).to(device)


# def get_correlate_matrix(datatype, data):
#     R = get_train_R_matrix(datatype, data)
#     tmp_dict = {}
#     for i in range(R.shape[0]):
#         print(i)
#         tmp = []
#         for j in range(R.shape[0]):
#             if i == j:
#                 continue
#             tmp.append(cosine_similarity(R[i], R[j]))
#         tmp_dict[i] = softmax(np.array(tmp))
#     c_matrix = np.zeros(shape=(data_params[datatype]['stu_num'], data_params[datatype]['stu_num']))
#     for k in range(c_matrix.shape[0]):
#         tmp_list = tmp_dict[k].tolist()
#         tmp_list.insert(k, 0)
#         c_matrix[k] = np.array(tmp_list)
#     return c_matrix


epochs_dict = {
    'Math2': {
        'ncdm': 3,
        'hiercdf': 1,
        'kancd': 1,
        'mirt': 8,
        'dina': 10,
        'kscd': 5

    },
    'Math1': {
        'ncdm': 3,
        'hiercdf': 1,
        'kancd': 1,
        'mirt': 15,
        'dina': 10,
        'kscd': 5
    }, 'junyi': {
        'ncdm': 5,
        'hiercdf': 5,
        'kancd': 1,
        'mirt': 25,
        'dina': 20,
        'kscd': 3,

    },
    'a910': {
        'ncdm': 5,
        'mirt': 15,
        'dina': 15,
        'kancd': 1,
        'kscd': 3
    },
    'a17': {
        'ncdm': 5,
        'mirt': 15,
        'dina': 15,
        'kancd': 1,
        'kscd': 8
    },
    'nips20': {
        'ncdm': 5,
        'mirt': 15,
        'dina': 15,
        'kancd': 1,
        'kscd': 3
    },
    'a09': {
        'ncdm': 5,
        'mirt': 15,
        'dina': 15,
        'kancd': 1
    },
    'FrcSub': {
        'ncdm': 5,
        'mirt': 15,
        'dina': 15,
        'kancd': 5,
        'kscd': 5
    },
    'EdNet-1':
        {
            'kancd':2,
            'ncdm': 9,
            'mirt': 15,
            'dina': 15,
            'kscd': 5,
        }
}

import os
import pickle
def save(config: dict, mas):
    exp_type = config['exp_type']
    method = config['method']
    name = config['name']
    if exp_type == 'gcn':
        name += '-' + str(config['gcn'])
    elif exp_type == 'dim':
        name += '-' + str(config['dim'])
    elif exp_type == 'dis':
        name += '-' + str(config['dis'])
    elif exp_type == 'sparse':
        name += '-' + str(config['new_ratio'])
    elif exp_type == 'khop':
        name += '-' + str(config['khop'])
    elif exp_type == 'agg':
        name += '-' + str(config['agg_type'])
    elif exp_type == 'reg':
        name += '-' + str(config['weight_reg'])
    if not os.path.exists(f'logs/{exp_type}/{method}'):
        os.makedirs(f'logs/{exp_type}/{method}')
    if exp_type == 'cdm' or exp_type == 'ind':
        mas_file_path = f'logs/{exp_type}/{method}' + '/' + name + '-Mas' + '.pkl'
        with open(mas_file_path, 'wb') as f:
            pickle.dump(mas, f)
    id_file_path = f'logs/{exp_type}/{method}' + '/' + name + '-id' + '.pkl'
    with open(id_file_path, 'wb') as f:
        pickle.dump(config['id'], f)