import numpy as np
import pickle

import pandas as pd


def mean_avg_distance(X):
    n = X.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = np.linalg.norm(X[i] - X[j], 2)
            dist[j, i] = dist[i, j]
    return np.mean(dist)

# datapath = '../exps/IGCDM/logs/cdm/'
# datatypes = ['nips20']
# for datatype in datatypes:
#     for method in ['igcdm-lightgcn', 'kancd', 'ncdm']:
#         mas_list = []
#         for i in range(10):
#             path = '{}/{}/{}-{}-seed{}-Mas.pkl'.format(datapath, method,
#                                                                                                   method,
#                                                                                                   datatype, i)
#             with open(path, 'rb') as f:
#                 mas = pickle.load(f)
#                 print(method, datatype, i, mean_avg_distance(mas[-1]))

