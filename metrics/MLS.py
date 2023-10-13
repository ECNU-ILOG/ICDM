import numpy as np
import pickle


def euclidean_distance(A, B, norm=False):
    if norm:
        return np.sqrt(np.sum((A - B) ** 2)) / np.sqrt(np.sum((np.ones(shape=(A.shape)) - np.zeros(shape=(B.shape))) ** 2))
    else:
        return np.sqrt(np.sum((A - B) ** 2))


def MLS(mas_list):
    mls = 0.0
    for i in range(len(mas_list)):
        for j in range(i + 1, len(mas_list)):
            mls += euclidean_distance(mas_list[i], mas_list[j])
    return 2 * mls / len(mas_list) / (len(mas_list) - 1)


for datatype in ['junyi', 'nips20', 'Math1', 'Math2']:
    for method in ['qccdm-softplus-mf', 'qccdm-softplus-single', 'qccdm-sigmoid-mf', 'qccdm-sigmoid-single',
                   'qccdm-tanh-mf', 'qccdm-tanh-single', 'kancd', 'ncdm']:
        mas_list = []
        for i in range(10):
            path = '/root/autodl-tmp/CDM-ILOG/exps/QCCDM/logs/cdm/{}/{}-{}-seed{}-Mas.pkl'.format(method,
                                                                                                  method,
                                                                                                  datatype, i)
            with open(path, 'rb') as f:
                mas = pickle.load(f)
                if 'qccdm' in method:
                    if datatype == 'junyi':
                        index = 3
                    else:
                        index = 2
                elif 'ncdm' in method:
                    if datatype == 'junyi':
                        index = -1
                    else:
                        index = 2
                else:
                    index = -1
                mas_list.append(mas[index])
        print(method, datatype, MLS(mas_list))
