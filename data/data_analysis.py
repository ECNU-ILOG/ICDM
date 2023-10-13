import pprint

import torch
import pickle
import pandas as pd
import numpy as np
from data_params_dict import data_params
import torch.nn.functional as F


def get_top_k_concepts(datatype: str, topk: int = 10):
    q = pd.read_csv('../data/{}/q.csv'.format(datatype), header=None).to_numpy()
    a = pd.read_csv('../data/{}/{}TotalData.csv'.format(datatype, datatype), header=None).to_numpy()
    skill_dict = {}
    for k in range(q.shape[1]):
        skill_dict[k] = 0
    for k in range(a.shape[0]):
        stu_id = a[k, 0]
        prob_id = a[k, 1]
        skills = np.where(q[int(prob_id), :] != 0)[0].tolist()
        for skill in skills:
            skill_dict[skill] += 1

    sorted_dict = dict(sorted(skill_dict.items(), key=lambda x: x[1], reverse=True))
    all_list = list(sorted_dict.keys()) # 189
    return all_list[:topk]


def get_average_correct_rate(datatype: str):
    data = pd.read_csv('../data/{}/{}TotalData.csv'.format(datatype, datatype), header=None).to_numpy()
    cdata = []
    right = 0
    for k in range(data.shape[0]):
        right += data[k, 2]
    return right / data.shape[0]


def get_q_density(datatype: str):
    q = pd.read_csv('../data/{}/q.csv'.format(datatype), header=None).to_numpy()
    return np.sum(q) / q.shape[0]


def get_group_by_correct_rate(datatype: str):
    students_dict = {}
    data = pd.read_csv('../data/{}/{}TotalData.csv'.format(datatype, datatype), header=None).to_numpy()
    for k in range(data.shape[0]):
        stu_id = data[k, 0]
        if students_dict.get(stu_id) is None:
            students_dict[stu_id] = 1.0
        else:
            students_dict[stu_id] += 1.0
    sorted_dict = dict(sorted(students_dict.items(), key=lambda x: x[1], reverse=True))
    keys = list(sorted_dict.keys())
    slices = len(keys) // 4
    high_indices = keys[:slices]
    middle_indices = keys[slices:slices * 3]
    low_indices = keys[slices * 3:]
    with open('../data/{}/{}high.pkl'.format(datatype, datatype), 'wb') as f:
        pickle.dump(high_indices, f)
    with open('../data/{}/{}middle.pkl'.format(datatype, datatype), 'wb') as f:
        pickle.dump(middle_indices, f)
    with open('../data/{}/{}low.pkl'.format(datatype, datatype), 'wb') as f:
        pickle.dump(low_indices, f)
    print('complete {}'.format(datatype))


def get_data_R_matrix(datatype: str):
    data = pd.read_csv('../data/{}/{}TotalData.csv'.format(datatype, datatype), header=None).to_numpy()
    stu_num = data_params[datatype]['stu_num']
    prob_num = data_params[datatype]['prob_num']
    R_matrix = np.zeros(shape=(stu_num, prob_num))
    for k in range(data.shape[0]):
        if data[k, 2] == 1:
            R_matrix[int(data[k, 0]), int(data[k, 1])] = 1
        else:
            R_matrix[int(data[k, 0]), int(data[k, 1])] = -1
    return R_matrix

def get_data_analysis(datatype:str):
    analysis_dict = {}
    data = pd.read_csv('../data/{}/{}TotalData.csv'.format(datatype, datatype), header=None).to_numpy()
    analysis_dict['number of responses'] = data.shape[0]
    stu_num = data_params[datatype]['stu_num']
    prob_num = data_params[datatype]['prob_num']
    know_num = data_params[datatype]['know_num']
    analysis_dict['datatype'] = datatype
    analysis_dict['stu_num'] = stu_num
    analysis_dict['prob_num'] = prob_num
    analysis_dict['know_num'] = know_num
    analysis_dict['sparsity'] = data.shape[0] / stu_num / prob_num
    analysis_dict['q-density'] = get_q_density(datatype)
    analysis_dict['acr'] = get_average_correct_rate(datatype)
    analysis_dict['topk_concepts'] = get_top_k_concepts(datatype, 10)
    pprint.pprint(analysis_dict)


get_data_analysis('EdNet-1')