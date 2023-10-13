import numpy as np

from method.Baselines.NCDM.ncdm import NCDM
from method.Baselines.MIRT.mirt import MIRT
from method.Baselines.DINA.dina import DINA
from method.Baselines.KANCD.kancd import KaNCD
from method.Baselines.KSCD.kscd import KSCD
from method.Baselines.RCD.rcd import RCD
from method.Baselines.Random.random import Random


def ncdm_runner(config, save):
    ncdm = NCDM(knowledge_n=config['know_num'], exer_n=config['prob_num'], student_n=config['stu_num'],
                device=config['device'])
    ncdm.train(np_train=config['np_train'], np_test=config['np_test'], epoch=config['epoch'], q=config['q'],
               batch_size=config['batch_size'])
    save(config, ncdm.mas_list)


def rcd_runner(config, save):
    rcd = RCD(config=config)
    rcd.train()
    save(config, rcd.mas_list)


def mirt_runner(config, save):
    mirt = MIRT(user_num=config['stu_num'], item_num=config['prob_num'], latent_dim=16, device=config['device'])
    mirt.train(config['np_train'], config['np_test'], epoch=config['epoch'], q=config['q'],
               batch_size=config['batch_size'])


def dina_runner(config, save):
    dina = DINA(user_num=config['stu_num'], item_num=config['prob_num'], hidden_dim=config['know_num'],
                device=config['device'])
    dina.train(np_train=config['np_train'], np_test=config['np_test'], epoch=config['epoch'], q=config['q'],
               batch_size=config['batch_size'])


def kancd_runner(config, save):
    kancd = KaNCD(stu_num=config['stu_num'], prob_num=config['prob_num'], know_num=config['know_num'],
                  device=config['device'])
    if config['exp_type'] == 'sparse':
        kancd.train(config['np_train'], config['np_test'], q=config['q'], batch_size=config['batch_size'],
                    epoch=config['epoch'], sp=True)
    else:
        kancd.train(config['np_train'], config['np_test'], q=config['q'], batch_size=config['batch_size'],
                    epoch=config['epoch'])
    save(config, kancd.mas_list)


def kscd_runner(config, save):
    kscd = KSCD(stu_num=config['stu_num'], prob_num=config['prob_num'], know_num=config['know_num'], dim=20,
                device=config['device'])
    kscd.train(config['np_train'], config['np_test'], q=config['q'], batch_size=config['batch_size'],
               epoch=config['epoch'])
    save(config, kscd.mas_list)


def kancd_re_ind_runner(config, save):
    kancd = KaNCD(stu_num=config['stu_num'], prob_num=config['prob_num'], know_num=config['know_num'],
                  device=config['device'], exist_idx=config['exist_idx'],
                  new_idx=config['new_idx'])
    kancd.train(np.vstack((config['np_train_old'], config['np_train_new'])), config['np_test'], config['np_test_new'],
                q=config['q'],
                batch_size=config['batch_size'], epoch=config['epoch'])
    save(config, kancd.mas_list)


def kancd_pos_ind_runner(config, save):
    kancd = KaNCD(stu_num=config['stu_num'], prob_num=config['prob_num'], know_num=config['know_num'],
                  device=config['device'], exist_idx=config['exist_idx'],
                  new_idx=config['new_idx'])
    kancd.train(np_train=config['np_train_old'], np_test=config['np_test'], np_test_new=config['np_test_new'],
                q=config['q'],
                batch_size=config['batch_size'], epoch=config['epoch'])
    kancd.pos_strategy_for_inductive_eval(np_train=config['np_train_old'], np_test=config['np_test'],
                                          np_test_new=config['np_test_new'], q=config['q'],
                                          batch_size=config['batch_size'])
    save(config, kancd.mas_list)


def kancd_closest_ind_runner(config, save):
    kancd = KaNCD(stu_num=config['stu_num'], prob_num=config['prob_num'], know_num=config['know_num'],
                  device=config['device'], exist_idx=config['exist_idx'],
                  new_idx=config['new_idx'])
    kancd.train(np_train=config['np_train_old'], np_test=config['np_test'], np_test_new=config['np_test_new'],
                q=config['q'],
                batch_size=config['batch_size'], epoch=config['epoch'])
    kancd.closest_strategy_for_inductive_eval(np_train_all=np.vstack((config['np_train_old'], config['np_train_new'])),
                                              np_test=config['np_test'],
                                              np_test_new=config['np_test_new'], q=config['q'],
                                              batch_size=config['batch_size'])
    save(config, kancd.mas_list)


def random_ind_runner(config, save):
    random = Random(stu_num=config['stu_num'], prob_num=config['prob_num'], know_num=config['know_num'],
                    device=config['device'], exist_idx=config['exist_idx'],
                    new_idx=config['new_idx'])
    random.train(np_train=config['np_train_old'], np_test=config['np_test'], np_test_new=config['np_test_new'],
                 q=config['q'],
                 batch_size=config['batch_size'], epoch=config['epoch'])
