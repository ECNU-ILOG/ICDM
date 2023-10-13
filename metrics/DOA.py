import numpy as np
import multiprocessing
from tqdm import tqdm
from joblib import Parallel, delayed


def calculate_doa_k_ood(mas_level, q_matrix, r_matrix, k):
    n_students, n_skills = mas_level.shape
    n_questions, _ = q_matrix.shape
    n_attempts = r_matrix.shape[1]
    DOA_k = 0.0
    numerator = 0
    denominator = 0
    delta_matrix = mas_level[:, k].reshape(-1, 1) > mas_level[:, k].reshape(1, -1)
    question_hask = np.where(q_matrix[:, k] != 0)[0].tolist()
    for j in question_hask:
        row_vector = (r_matrix[:, j].reshape(1, -1) != -1).astype(int)
        columen_vector = (r_matrix[:, j].reshape(-1, 1) != -1).astype(int)
        mask = row_vector * columen_vector
        delta_r_matrix = r_matrix[:, j].reshape(-1, 1) > r_matrix[:, j].reshape(1, -1)
        I_matrix = r_matrix[:, j].reshape(-1, 1) != r_matrix[:, j].reshape(1, -1)
        numerator_ = np.logical_and(mask, delta_r_matrix)
        denominator_ = np.logical_and(mask, I_matrix)
        numerator += np.sum(delta_matrix * numerator_)
        denominator += np.sum(delta_matrix * denominator_)
    if denominator != 0:
        DOA_k = numerator / denominator
        return [k, DOA_k]
    else:
        return [k, 0]


# def calculate_doa_k(mastery_level, q_matrix, r_matrix, k):
#     student_n = mastery_level.shape[0]
#     prob_n = q_matrix.shape[0]
#     doa_k = 0
#     z = 0
#     delta_m_indices = np.nonzero(np.diff(mastery_level[:, k], axis=0))[0]
#     for i in range(len(delta_m_indices)):
#         a = delta_m_indices[i]
#         for b in range(a + 1, student_n):
#             delta_m = int(mastery_level[a, k] > mastery_level[b, k])
#             if delta_m == 0:
#                 continue
#             mask = np.logical_and(r_matrix[a] != -1, r_matrix[b] != -1)
#             J_a_b = np.ones(prob_n)
#             I_a_b = np.where(r_matrix[a, mask] != r_matrix[b, mask], 1, 0)
#             delta_r = np.where(r_matrix[a, mask] > r_matrix[b, mask], 1, 0)
#             numerator = np.sum(q_matrix[mask, k] * J_a_b[mask] * delta_r)
#             denominator = np.sum(q_matrix[mask, k] * J_a_b[mask] * I_a_b)
#             if denominator != 0:
#                 doa_k += delta_m * numerator / denominator
#                 z += delta_m
#     print(k, 'compeleted', doa_k / z)
#     if z == 0:
#         return 0
#     else:
#         return doa_k/z
def calculate_doa_k(mas_level, q_matrix, r_matrix, k):
    n_students, n_skills = mas_level.shape
    n_questions, _ = q_matrix.shape
    n_attempts = r_matrix.shape[1]
    DOA_k = 0.0
    numerator = 0
    denominator = 0
    delta_matrix = mas_level[:, k].reshape(-1, 1) > mas_level[:, k].reshape(1, -1)
    question_hask = np.where(q_matrix[:, k] != 0)[0].tolist()
    for j in question_hask:
        row_vector = (r_matrix[:, j].reshape(1, -1) != -1).astype(int)
        column_vector = (r_matrix[:, j].reshape(-1, 1) != -1).astype(int)
        mask = row_vector * column_vector
        delta_r_matrix = r_matrix[:, j].reshape(-1, 1) > r_matrix[:, j].reshape(1, -1)
        I_matrix = r_matrix[:, j].reshape(-1, 1) != r_matrix[:, j].reshape(1, -1)
        numerator_ = np.logical_and(mask, delta_r_matrix)
        denominator_ = np.logical_and(mask, I_matrix)
        numerator += np.sum(delta_matrix * numerator_)
        denominator += np.sum(delta_matrix * denominator_)

    if denominator != 0:
        DOA_k = numerator / denominator
        return DOA_k
    else:
        return 0


def calculate_doa_k_block(mas_level, q_matrix, r_matrix, k, block_size=50):
    n_students, n_skills = mas_level.shape
    n_questions, _ = q_matrix.shape
    n_attempts = r_matrix.shape[1]
    DOA_k = 0.0
    numerator = 0
    denominator = 0
    question_hask = np.where(q_matrix[:, k] != 0)[0].tolist()
    for start in range(0, n_students, block_size):
        end = min(start + block_size, n_students)
        mas_level_block = mas_level[start:end, :]
        delta_matrix_block = mas_level[start:end, k].reshape(-1, 1) > mas_level[start:end, k].reshape(1, -1)
        r_matrix_block = r_matrix[start:end, :]
        for j in question_hask:
            row_vector = (r_matrix_block[:, j].reshape(1, -1) != -1).astype(int)
            columen_vector = (r_matrix_block[:, j].reshape(-1, 1) != -1).astype(int)
            mask = row_vector * columen_vector
            delta_r_matrix = r_matrix_block[:, j].reshape(-1, 1) > r_matrix_block[:, j].reshape(1, -1)
            I_matrix = r_matrix_block[:, j].reshape(-1, 1) != r_matrix_block[:, j].reshape(1, -1)
            numerator_ = np.logical_and(mask, delta_r_matrix)
            denominator_ = np.logical_and(mask, I_matrix)
            numerator += np.sum(delta_matrix_block * numerator_)
            denominator += np.sum(delta_matrix_block * denominator_)
    if denominator == 0:
        DOA_k = 0
    else:
        DOA_k = numerator / denominator
    return DOA_k


def DOA(mastery_level, q_matrix, r_matrix):
    know_n = q_matrix.shape[1]
    doa_k_list = Parallel(n_jobs=-1)(
        delayed(calculate_doa_k)(mastery_level, q_matrix, r_matrix, k) for k in range(know_n))
    return np.mean(doa_k_list)


def DOA_Junyi(mastery_level, q_matrix, r_matrix, concepts=None):
    if concepts is None:
        concepts = [433, 28, 653, 563, 631, 392, 632, 393, 652, 394]
    know_n = q_matrix.shape[1]
    # concepts = np.random.randint(0, know_n, 20)
    doa_k_list = Parallel(n_jobs=-1)(
        delayed(calculate_doa_k_block)(mastery_level, q_matrix, r_matrix, k, 2000) for k in concepts)
    return np.mean(doa_k_list)


def DOA_Junyi835(mastery_level, q_matrix, r_matrix, concepts=None):
    if concepts is None:
        concepts = [487, 31, 749, 633, 727, 442, 728, 443, 748, 32]
    know_n = q_matrix.shape[1]
    # concepts = np.random.randint(0, know_n, 20)
    doa_k_list = Parallel(n_jobs=-1)(
        delayed(calculate_doa_k_block)(mastery_level, q_matrix, r_matrix, k, 2000) for k in concepts)
    return np.mean(doa_k_list)


def DOA_Assist910(mastery_level, q_matrix, r_matrix):
    concepts = [98, 30, 79, 82, 49, 99, 32, 81, 45, 6]
    doa_k_list = Parallel(n_jobs=-1)(
        delayed(calculate_doa_k_block)(mastery_level, q_matrix, r_matrix, k) for k in concepts)
    return np.mean(doa_k_list)


def DOA_Assist17(mastery_level, q_matrix, r_matrix):
    concepts = [21, 58, 14, 5, 33, 34, 10, 7, 4, 60]
    doa_k_list = Parallel(n_jobs=-1)(
        delayed(calculate_doa_k_block)(mastery_level, q_matrix, r_matrix, k) for k in concepts)
    return np.mean(doa_k_list)


def DOA_OOD(mastery_level, q_matrix, r_matrix):
    know_n = q_matrix.shape[1]
    doa_k_list = Parallel(n_jobs=-1)(
        delayed(calculate_doa_k_ood)(mastery_level, q_matrix, r_matrix, k) for k in range(know_n))
    return doa_k_list


def DOA_Nips20(mastery_level, q_matrix, r_matrix):
    concepts = [0, 1, 17, 38, 87, 8, 67, 91, 9, 30]
    # concepts = [0, 1, 36, 16, 7, 78, 62, 39, 77, 82]
    doa_k_list = Parallel(n_jobs=-1)(
        delayed(calculate_doa_k_block)(mastery_level, q_matrix, r_matrix, k) for k in concepts)
    return np.mean(doa_k_list)


def DOA_Assist09(mastery_level, q_matrix, r_matrix):
    concepts = [82, 23, 63, 66, 35, 39, 26, 9, 83, 10]
    doa_k_list = Parallel(n_jobs=-1)(
        delayed(calculate_doa_k_block)(mastery_level, q_matrix, r_matrix, k) for k in concepts)
    return np.mean(doa_k_list)


def DOA_EdNet_1(mastery_level, q_matrix, r_matrix):
    concepts = [181, 179, 180, 182, 177, 183, 24, 52, 2, 30]
    doa_k_list = Parallel(n_jobs=-1)(
        delayed(calculate_doa_k_block)(mastery_level, q_matrix, r_matrix, k) for k in concepts)
    return np.mean(doa_k_list)
