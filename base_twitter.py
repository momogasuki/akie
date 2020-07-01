import torch
import torch.nn.functional as F
from torch import nn, optim

import numpy as np
from numpy import *

import math, pickle
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from akie import BaseLearner, setup_seed

def get_food(emb, user, ui_u, phase, tot_user=296079):
    if phase == 'train':
        itrue = ui_u['T']
        ui_u['F'] = np.random.choice(np.setdiff1d(np.arange(tot_user), np.concatenate((itrue, [user]))), len(itrue), False)
    onehot_i = torch.rand(2*len(ui_u['T']), 0).long().cuda()
    onehot_x = torch.rand(2*len(ui_u['T']), 0).double().cuda()
    multihot_list = []
    ctns_i = torch.tensor(np.concatenate((emb[ui_u['T']], emb[ui_u['F']]))).cuda().double()
    ctns_u = torch.tensor(emb[user]).repeat(ctns_i.shape[0], 1).cuda().double()
    ctns = torch.cat((ctns_u, ctns_i), 1)
    label = torch.cat((torch.ones(ui_u['T'].shape), torch.zeros(ui_u['F'].shape))).cuda().double()
    return onehot_i, onehot_x, multihot_list, ctns, label

def get_valid_test_food(emb, ui_valid, ui_test):
    onehot_is_valid, onehot_xs_valid, ctnss_valid, labels_valid = [], [], [], []
    for user in ui_valid.keys():
        onehot_i, onehot_x, multihot_list, ctns, label = get_food(emb, user, ui_valid[user], 'valid')
        onehot_is_valid.append(onehot_i)
        onehot_xs_valid.append(onehot_x)
        ctnss_valid.append(ctns)
        labels_valid.append(label)
    onehot_i_valid = torch.cat(onehot_is_valid)
    onehot_x_valid = torch.cat(onehot_xs_valid)
    ctns_valid = torch.cat(ctnss_valid)
    label_valid = torch.cat(labels_valid)

    onehot_is_test, onehot_xs_test, ctnss_test, labels_test = [], [], [], []
    for user in ui_test.keys():
        onehot_i, onehot_x, multihot_list, ctns, label = get_food(emb, user, ui_test[user], 'test')
        onehot_is_test.append(onehot_i)
        onehot_xs_test.append(onehot_x)
        ctnss_test.append(ctns)
        labels_test.append(label)
    onehot_i_test = torch.cat(onehot_is_test)
    onehot_x_test = torch.cat(onehot_xs_test)
    ctns_test = torch.cat(ctnss_test)
    label_test = torch.cat(labels_test)

    return onehot_i_valid, onehot_x_valid, ctns_valid, label_valid, onehot_i_test, onehot_x_test, ctns_test, label_test
    
def main(config):
    with open("/home2/zh/data/twitter/train_valid_test.pkl", 'rb') as f:
        ui_train, ui_valid, ui_test = pickle.load(f)
    emb = np.load("/home2/zh/data/twitter/emb.npy")

    onehot_i_valid, onehot_x_valid, ctns_valid, label_valid, onehot_i_test, onehot_x_test, ctns_test, label_test = get_valid_test_food(emb, ui_valid, ui_test)

    setup_seed(config['seed'])
    model = BaseLearner(config).cuda().double()
    optimizer = optim.SGD([{'params': list(model.parameters())[0], 'lr': config['lr'][0]},
                            {'params': list(model.parameters())[1:], 'lr': config['lr'][1]}], lr=1)
    sample_cnt = 0
    vaaucs = []
    maxauc = 0
    bestmodeldic = None
    for _ in range(5000):
        losses, reses, labels = [], [], []
        for T_T in range(32):
            user = np.random.choice(list(ui_train.keys()), 1).item()
            onehot_i, onehot_x, multihot_list, ctns, label = get_food(emb, user, ui_train[user], 'train')
            res = model(onehot_i, onehot_x, multihot_list, ctns)
            loss = model.loss_func(res, label)
            losses.append(loss)
            reses.append(res)
            labels.append(label)
            sample_cnt += len(label)
        optimizer.zero_grad()
        # if _ == 8:
        #     from IPython import embed; embed(); exit()
        torch.stack(losses).mean().backward()
        optimizer.step()
        if _ % 10 == 0:
            print('---- batch {} ---- sample_cnt: {}'.format(_, sample_cnt))
            # print(list(model.parameters())[1][0,:10])
            # print(list(model.parameters())[2][:10])
            # print(list(model.parameters())[3][0,:10])
            # print(list(model.parameters())[4])
            # sample_cnt = 0
            # from IPython import embed; embed(); exit()
            # from IPython import embed; embed(); exit()
            model.eval()
            res = model(onehot_i_valid, onehot_x_valid, [], ctns_valid)
            model.train()
            loss = model.loss_func(res, label_valid)
            # print(roc_auc_score(array(torch.cat(labels).detach().cpu()), array(torch.cat(reses).detach().cpu())))
            # print(torch.stack(losses).mean())
            # print(torch.cat(labels).shape)
            auc = roc_auc_score(array(label_valid.detach().cpu()), array(res.detach().cpu()))
            vaaucs.append(auc)
            if auc > maxauc:
                bestmodeldic = model.state_dict()
                maxauc = auc
            # print(loss.mean())
            # print(label.shape)
            print('maxauc = {}'.format(maxauc))
            for pg in optimizer.param_groups:
                if pg['lr'] > config['decay'][0]:
                    pg['lr'] *= config['decay'][1]
    # torch.save(model.state_dict(), 'try.wddp')
    pkl_file = config['pkl_file']
    try:
        with open(pkl_file, 'rb') as f:
            logg = pickle.load(f)
    except:
        logg = []
    logg_this_try = [config, vaaucs]
    if bestmodeldic:
        model.load_state_dict(bestmodeldic)
        model.eval()
        res = model(onehot_i_test, onehot_x_test, [], ctns_test)
        model.train()
        loss = model.loss_func(res, label_test).item()
        auc = roc_auc_score(array(label_test.detach().cpu()), array(res.detach().cpu()))
        logg_this_try += [auc, loss]
        print('best valid auc = {}, test auc = {}'.format(maxauc, auc))
    logg.append(logg_this_try)
    with open(pkl_file, 'wb') as f:
        pickle.dump(logg, f)

if __name__ == '__main__':
    torch.cuda.set_device(1)
    # config = {'num_embeddings': 3529,
    #         'embedding_dim': 32,
    #         'dim': [32*6+1, 128, 64, 1],
    #         'lr': [0.001, 1],
    #         'dropout': [0, 0, 0.5],
    #         'decay': [0, 1],
    #         'leaky': 0,
    #         'pkl_file': 'res_wddp.pkl',
    #         'seed': 81192}
    config = {'num_embeddings': 2*768,
            'embedding_dim': 16,
            'dim': [16*0+768*2, 256, 1],
            'lr': [0.01, 1],
            'dropout': [0, 0],
            'decay': [0.01, 0.98],
            'seed': 81192,
            'pkl_file': 'res_base_twitter.pkl'}
    for lr in [[0.001, 0.001], [0.005, 0.5]]:
        for dropout in [[0, 0, 0], [0.1, 0, 0], [0, 0, 0.5], [0.1, 0.1, 0.1]]:
            for decay in [[0, 1]]:
                config['lr'] = lr
                config['dropout'] = dropout
                config['decay'] = decay
                print(config)
                main(config)