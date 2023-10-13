from method.IGCDM.igcdm import IGCDM
from method.IGCDM.igcdm_ind import IGCDM as IGCDMIND
from method.Baselines.IMCGAE.imcgae import IMCGAE
from runners.commonutils.commonrunners import  *
from runners.IGCDM.utils import save
import numpy as np

def igcdm_runner(config, save):
    igcdm = IGCDM(stu_num=config['stu_num'], prob_num=config['prob_num'], know_num=config['know_num'],
                  dim=config['dim'], device=config['device'], gcn_layers=config['gcnlayers'],
                  weight_reg=config['weight_reg'],
                  graph=config['graph_dict'], agg_type=config['agg_type'], cdm_type=config['cdm_type'],
                  khop=config['khop'])
    igcdm.train(config['np_train'], config['np_test'], q=config['q'], batch_size=config['batch_size'],
                epoch=config['epoch'], lr=config['lr'])
    save(config, igcdm.mas_list)


def get_runner(method: str):
    if 'igcdm' in method:
        return igcdm_runner
    elif 'kancd' in method:
        return kancd_runner
    elif 'ncdm' in method:
        return ncdm_runner
    elif 'rcd' in method:
        return rcd_runner
    elif 'mirt' in method:
        return mirt_runner
    elif 'kscd' in method:
        return kscd_runner
    elif 'dina' in method:
        return dina_runner
    else:
        raise ValueError('This method is currently not supported.')


def igcdm_ind_runner(config, save):
    if config['ab'] == 'tf':
        config['dim'] = config['know_num']
    igcdm = IGCDMIND(stu_num=config['stu_num'], prob_num=config['prob_num'], know_num=config['know_num'],
                     dim=config['dim'], device=config['device'], gcn_layers=config['gcnlayers'],
                     weight_reg=config['weight_reg'],
                     graph=config['graph_dict'], agg_type=config['agg_type'], exist_idx=config['exist_idx'],
                     new_index=config['new_idx'], mode=config['mode'], cdm_type=config['cdm_type'], khop=config['khop'],
                     ab=config['ab'], d_1=config['d_1'], d_2=config['d_2'])
    igcdm.train(config['np_train_old'], config['np_train_new'], config['np_test'], config['np_test_new'], q=config['q'],
                batch_size=config['batch_size'],
                epoch=config['epoch'], lr=config['lr'])
    save(config, igcdm.mas_list)


def imcgae_ind_runer(config, save):
    imcgae = IMCGAE(stu_num=config['stu_num'], prob_num=config['prob_num'], know_num=config['know_num'],
                    dim=config['dim'], device=config['device'], gcn_layers=config['gcnlayers'], mode=config['mode'],
                    exist_idx=config['exist_idx'],
                    new_index=config['new_idx'], cdm_type=config['cdm_type'])
    imcgae.train(config['np_train_old'], config['np_train_new'], config['np_test'], config['np_test_new'],
                 q=config['q'],
                 batch_size=config['batch_size'],
                 epoch=config['epoch'], lr=config['lr'])
    save(config, imcgae.mas_list)


def igcdm_re_ind_runner(config, save):
    igcdm = IGCDM(stu_num=config['stu_num'], prob_num=config['prob_num'], know_num=config['know_num'],
                  dim=config['dim'], device=config['device'], gcn_layers=config['gcnlayers'],
                  weight_reg=config['weight_reg'],
                  graph=config['graph_dict'], agg_type=config['agg_type'], exist_idx=config['exist_idx'],
                  new_idx=config['new_idx'], cdm_type=config['cdm_type'], khop=config['khop'])
    igcdm.train(np.vstack((config['np_train_old'], config['np_train_new'])), config['np_test'], config['np_test_new'], q=config['q'],
                batch_size=config['batch_size'],
                epoch=config['epoch'], lr=config['lr'])
    save(config, igcdm.mas_list)


def get_ind_runner(method: str):
    if 'igcdm' in method:
        if 'igcdm-re' not in method:
            return igcdm_ind_runner
        else:
            return igcdm_re_ind_runner
    elif 'imcgae' in method:
        return imcgae_ind_runer
    elif 'kancd-re' in method:
        return kancd_re_ind_runner
    elif 'kancd-pos' in method:
        return kancd_pos_ind_runner
    elif 'kancd-closest' in method:
        return kancd_closest_ind_runner
    elif 'random' in method:
        return random_ind_runner
    else:
        raise ValueError('This method is currently not supported.')
