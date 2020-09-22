from akie import MetaLearner

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn import ModuleList as ML
from torch.nn import ParameterList as PL
import numpy as np

import time, pickle, argparse
from sklearn.metrics import roc_auc_score

from util import setup_seed

def get_data():
    i_genre = np.load('/home2/zh/data/ml-1m/i_genre.npy')
    i_other = np.load('/home2/zh/data/ml-1m/i_other.npy')
    ui = np.load('/home2/zh/data/ml-1m/ui.npy')
    x_genre = np.load('/home2/zh/data/ml-1m/x_genre.npy')
    x_other = np.load('/home2/zh/data/ml-1m/x_other.npy')
    y = np.load('/home2/zh/data/ml-1m/y.npy')
    return i_genre, i_other, ui, x_genre, x_other, y

def run0(model, data, user, phase, spt_qry_split=None):
    i_genre, i_other, ui, x_genre, x_other, y = data
    idx = np.where(ui[:,0]==user)[0]
    if phase == 'train':
        if len(idx) <= 10: return None
    onehot_i = torch.tensor(i_other[idx,:-1]).cuda()
    onehot_x = torch.tensor(x_other[idx,:-1]).cuda()
    multihot_i = torch.tensor(i_genre[idx]).cuda()
    multihot_x = torch.tensor(x_genre[idx]).cuda()
    multihot_list = [(multihot_i, multihot_x)]
    ctns = torch.tensor(x_other[idx,-1:]).cuda()
    task_data = (onehot_i, onehot_x, multihot_list, ctns)
    label = torch.tensor(y[idx]).cuda().double()
    spt_qry_perm = torch.randperm(len(idx))
    if phase == 'train':
        if spt_qry_split == 'max(1/8, 4)':
            sqsplit = max(4, int(len(idx)/8))
        else:
            raise Exception('Undeifined spt_qry_split')
        spt_idx = spt_qry_perm[:sqsplit]
        qry_idx = spt_qry_perm[-sqsplit:]
        if (1-label[spt_idx]).sum() <= 0: return None
        if (1-label[qry_idx]).sum() <= 0: return None
    else:
        spt_idx = spt_qry_perm[:0]
        qry_idx = spt_qry_perm[:]
        model.eval()
    pred, loss = model(task_data, label, spt_idx, qry_idx, phase=phase)
    if phase != 'train':
        model.train()
    return pred, loss, label[qry_idx]

if __name__ == '__main__':
    torch.cuda.set_device(6)
    data = get_data()
    i_genre, i_other, ui, x_genre, x_other, y = data
    # config = {'num_embeddings': 3529,
    #         'embedding_dim': 32,
    #         'dim': [32*6+1, 64, 1],
    #         'dropout': [0, 0],
    #         'embedding_dim_meta': 32,
    #         'userl2': 32*4+0,
    #         'cluster_d': 32,
    #         'clusternum': [1, 3, 2, 1],
    #         'user_part': ([0,1,2,3], [], []),
    #         'inner_steps': 3,
    #         'batchsize': 256,
    #         'learning_rate': [0.5, 0.5],
    #         'update_lr': [0.05, (0, 0)],
    #         'seed': 81192}
    config = {'num_embeddings': 3529,
            'num_ctns': 1,
            'fieldnum': 7,
            'embedding_dim': 32,
            'headnum': 8,
            'attention_dim': 64,
            'embedding_dim_meta': 32,
            'userl2': 32*4+0,
            'cluster_d': 32,
            'clusternum': [1, 3, 2, 1],
            'user_part': ([0,1,2,3], [], []),
            'inner_steps': 3,
            'batchsize': 32,
            'learning_rate': [0.07, 0.7],
            'update_lr': [0.05, (0, 0)],
            'seed': 81192}
    config['spt_qry_split'] = 'max(1/8, 4)'
    config['batchnum'] = int(75000*32/config['batchsize'])
    config['eval_every'] = int(config['batchnum']/400)
    setup_seed(config['seed'])
    model = MetaLearner(config).cuda().double()
    user_set = np.array(list(set(ui[:,0])))
    usernum = len(user_set)
    perm = np.random.permutation(usernum)
    train_usernum = int(0.8*usernum)
    valid_usernum = int(0.9*usernum)
    optimizer = optim.SGD([{'params': model.net.parameters(), 'lr': config['learning_rate'][0]},
                            {'params': list(model.parameters())[len(list(model.net.parameters())):], 'lr': config['learning_rate'][1]}])
    traucs, vaaucs, trloss, valoss = [], [], [], []
    labels_chkp, preds_chkp, losses_chkp = [], [], []
    maxauc = 0
    test_auc, test_loss = 0, 0
    bestmodeldic = None
    for _ in range(config['batchnum']):
        preds, losses, labels = [], [], []
        for T_T in range(config['batchsize']):
            user = user_set[perm[np.random.randint(0,train_usernum)]]
            ret = run0(model, data, user, 'train', config['spt_qry_split'])
            if ret:
                pred, loss, label = ret
                preds.append(pred)
                losses.append(loss)
                labels.append(label)
        preds_chkp += preds
        losses_chkp += losses
        labels_chkp += labels
        if losses:
            optimizer.zero_grad()
            torch.stack(losses).mean().backward()
            optimizer.step()
        if _ % config['eval_every'] == 0:
            print('---- batch {} ----'.format(_))
            train_auc = roc_auc_score(np.array(torch.cat(labels_chkp).detach().cpu()), np.array(torch.cat(preds_chkp).detach().cpu()))
            train_loss = torch.stack(losses_chkp).mean().item()
            print('train_auc = {}, train_loss = {}'.format(train_auc, train_loss))
            labels_chkp, preds_chkp, losses_chkp = [], [], []
            traucs.append(train_auc)
            trloss.append(train_loss)
            preds, losses, labels = [], [], []
            for user in user_set[perm[train_usernum:valid_usernum]]:
                with torch.no_grad():
                    pred, loss, label = run0(model, data, user, 'valid')
                preds.append(pred)
                losses.append(loss)
                labels.append(label)
            valid_auc = roc_auc_score(np.array(torch.cat(labels).detach().cpu()), np.array(torch.cat(preds).detach().cpu()))
            valid_loss = torch.stack(losses).mean().item()
            print('valid_auc = {}, valid_loss = {}'.format(valid_auc, valid_loss))
            vaaucs.append(valid_auc)
            valoss.append(valid_loss)

            if valid_auc > maxauc:
                bestmodeldic = model.state_dict()
                maxauc = valid_auc

                if maxauc > 0.73:
                    preds, losses, labels = [], [], []
                    for user in user_set[perm[valid_usernum:]]:
                        with torch.no_grad():
                            pred, loss, label = run0(model, data, user, 'test')
                        preds.append(pred)
                        losses.append(loss)
                        labels.append(label)
                    test_auc = roc_auc_score(np.array(torch.cat(labels).detach().cpu()), np.array(torch.cat(preds).detach().cpu()))
                    test_loss = torch.stack(losses).mean().item()
            
            print('maxauc = {}, test_auc = {}'.format(maxauc, test_auc))
    
    pkl_file = 'res.pkl'
    try:
        with open(pkl_file, 'rb') as f:
            res = pickle.load(f)
    except:
        res = []
    
    res_this_try = [config, traucs, trloss, vaaucs, valoss]

    if test_auc < 0.761:
        bestmodeldic = None
    res_this_try += [bestmodeldic, test_auc, test_loss]
    
    res.append(res_this_try)
    with open(pkl_file, 'wb') as f:
        pickle.dump(res, f)