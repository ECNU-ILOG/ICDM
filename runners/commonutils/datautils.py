import pprint

import torch
import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from metrics.DOA import *
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from data.data_params_dict import data_params


def transform(q: torch.tensor, user, item, score, batch_size, dtype=torch.float64):
    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64),
        torch.tensor(item, dtype=torch.int64),
        q[item, :],
        torch.tensor(score, dtype=dtype)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)
def get_r_matrix_for_closest(np_test, stu_num, prob_num):
    r = np.zeros(shape=(stu_num, prob_num))

    for i in range(np_test.shape[0]):
        s = int(np_test[i, 0])
        p = int(np_test[i, 1])
        score = np_test[i, 2]
        if int(score) == 0:
            r[s, p] = -1
        else:
            r[s, p] = int(score)
    return r


def get_r_matrix(np_test, stu_num, prob_num, new_idx=None):
    if new_idx is None:
        r = -1 * np.ones(shape=(stu_num, prob_num))
        for i in range(np_test.shape[0]):
            s = int(np_test[i, 0])
            p = int(np_test[i, 1])
            score = np_test[i, 2]
            r[s, p] = int(score)
    else:
        r = -1 * np.ones(shape=(stu_num, prob_num))

        for i in range(np_test.shape[0]):
            s = new_idx.index(int(np_test[i, 0]))
            p = int(np_test[i, 1])
            score = np_test[i, 2]
            r[s, p] = int(score)
    return r


def get_doa_function(know_num):
    if know_num == 734:
        doa_func = DOA_Junyi
    elif know_num == 835:
        doa_func = DOA_Junyi835
    elif know_num == 123:
        doa_func = DOA_Assist910
    elif know_num == 102:
        doa_func = DOA_Assist17
    elif know_num == 268:
        doa_func = DOA_Nips20
    elif know_num == 95:
        doa_func = DOA_Assist09
    elif know_num == 189:
        doa_func = DOA_EdNet_1
    else:
        doa_func = DOA
    return doa_func


def get_group_acc(know_num):
    if know_num == 734:
        datatype = 'junyi'
    elif know_num == 123:
        datatype = 'a910'
    elif know_num == 102:
        datatype = 'a17'
    elif know_num == 268:
        datatype = 'nips20'
    with open('../../data/{}/{}high.pkl'.format(datatype, datatype), 'rb') as f:
        high = pickle.load(f)
    with open('../../data/{}/{}middle.pkl'.format(datatype, datatype), 'rb') as f:
        middle = pickle.load(f)
    with open('../../data/{}/{}low.pkl'.format(datatype, datatype), 'rb') as f:
        low = pickle.load(f)
    return high, middle, low

# def get_split_by_student(datatype: str, new_student_ratio=0.2, test_ratio=0.2, seed=0):
#     data = pd.read_csv('../../data/{}/{}TotalData.csv'.format(datatype, datatype), header=None,
#                        names=['stu_id', 'prob_id', 'score'])
#     stu_num = data_params[datatype]['stu_num']
#     stu_idx = [i for i in range(stu_num)]
#     exist_stu_idx = np.random.choice(stu_idx, size=int((1 - new_student_ratio) * stu_num), replace=False)
#     new_stu_idx = [i for i in stu_idx if i not in exist_stu_idx]
#     exist_stu_data = data[data['stu_id'].isin(exist_stu_idx)]
#     new_stu_data = data[data['stu_id'].isin(new_stu_idx)]
#     exist_train = exist_stu_data.to_numpy()
#     new_data = new_stu_data.to_numpy()
#     new_train, new_test = train_test_split(new_data, test_size=test_ratio, random_state=seed)
#     np_train_full = np.vstack((exist_train, new_train))
#     np_test = new_test
#     exist_stu_idx = [int(i) for i in exist_stu_idx]
#     new_stu_idx = [int(i) for i in new_stu_idx]
#     return np_train_full, exist_train, np_test, exist_stu_idx, new_stu_idx
def get_split_by_student(datatype: str, new_student_ratio=0.2):
    data = pd.read_csv('../../data/{}/{}TotalData.csv'.format(datatype, datatype), header=None,
                       names=['stu_id', 'prob_id', 'score'])
    stu_num = data_params[datatype]['stu_num']
    stu_idx = [i for i in range(stu_num)]
    exist_stu_idx = np.random.choice(stu_idx, size=int((1 - new_student_ratio) * stu_num), replace=False)
    new_stu_idx = [i for i in stu_idx if i not in exist_stu_idx]
    exist_stu_data = data[data['stu_id'].isin(exist_stu_idx)]
    new_stu_data = data[data['stu_id'].isin(new_stu_idx)]
    exist_train = exist_stu_data.to_numpy()
    new_data = new_stu_data.to_numpy()
    np_train = exist_train
    np_test = new_data
    exist_stu_idx = [int(i) for i in exist_stu_idx]
    new_stu_idx = [int(i) for i in new_stu_idx]
    return np_train, np_test, exist_stu_idx, new_stu_idx


def get_split_inductive(datatype: str, test_size=0.2, new_student_ratio=0.2, seed=0):
    pd_data = pd.read_csv('../../data/{}/{}TotalData.csv'.format(datatype, datatype), header=None,
                          names=['stu_id', 'prob_id', 'score'])
    pd_train, pd_test = train_test_split(pd_data, test_size=test_size, random_state=seed)
    stu_num = data_params[datatype]['stu_num']
    stu_idx = [i for i in range(stu_num)]
    exist_stu_idx = np.random.choice(stu_idx, size=int((1 - new_student_ratio) * stu_num), replace=False)
    new_stu_idx = [i for i in stu_idx if i not in exist_stu_idx]

    exist_stu_data = pd_train[pd_train['stu_id'].isin(exist_stu_idx)]

    new_stu_data = pd_train[pd_train['stu_id'].isin(new_stu_idx)]

    np_test_all = pd_test.to_numpy()
    np_test_new = pd_test[pd_test['stu_id'].isin(new_stu_idx)]

    np_train_old = exist_stu_data.to_numpy()
    np_train_new = new_stu_data.to_numpy()
    exist_stu_idx = [int(i) for i in exist_stu_idx]
    new_stu_idx = [int(i) for i in new_stu_idx]

    return np_train_old, np_train_new, np_test_all, np_test_new.to_numpy(), exist_stu_idx, new_stu_idx




def get_split_by_student_acr(datatype: str, new_student_ratio=0.2, seed=0, known_dis='high'):
    data = pd.read_csv('../../data/{}/{}TotalData.csv'.format(datatype, datatype), header=None,
                       names=['stu_id', 'prob_id', 'score'])
    stu_num = data_params[datatype]['stu_num']
    stu_idx = [i for i in range(stu_num)]
    exist_stu_idx = np.random.choice(stu_idx, size=int((1 - new_student_ratio) * stu_num), replace=False)
    new_stu_idx = [i for i in stu_idx if i not in exist_stu_idx]
    exist_stu_data = data[data['stu_id'].isin(exist_stu_idx)]
    new_stu_data = data[data['stu_id'].isin(new_stu_idx)]
    exist_train = exist_stu_data.to_numpy()
    new_data = new_stu_data.to_numpy()

    data_np = exist_train
    np_test = new_data

    students_dict = {}
    for k in range(data_np.shape[0]):
        stu_id = data_np[k, 0]
        if students_dict.get(stu_id) is None:
            students_dict[stu_id] = 1.0
        else:
            students_dict[stu_id] += 1.0

    sorted_dict = dict(sorted(students_dict.items(), key=lambda x: x[1], reverse=True))
    keys = list(sorted_dict.keys())
    slices = len(keys) // 3
    high_indices = keys[:slices]
    middle_indices = keys[slices:slices * 2]
    low_indices = keys[slices * 2:]

    if known_dis == 'high':
        exist_stu_idx = middle_indices + low_indices
        exist_stu_data = data[data['stu_id'].isin(exist_stu_idx)]
    elif known_dis == 'middle':
        exist_stu_idx = high_indices + low_indices
        exist_stu_data = data[data['stu_id'].isin(exist_stu_idx)]
    else:
        exist_stu_idx = high_indices + middle_indices
        exist_stu_data = data[data['stu_id'].isin(exist_stu_idx)]

    np_train = exist_stu_data.to_numpy()
    exist_stu_idx = [int(i) for i in exist_stu_idx]
    return np_train, np_test, exist_stu_idx, new_stu_idx
