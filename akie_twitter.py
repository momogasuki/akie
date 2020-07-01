from akie import MetaLearner, setup_seed

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn import ModuleList as ML
from torch.nn import ParameterList as PL
import numpy as np

import time, pickle
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

def get_data():
    with open("/home2/zh/data/twitter/train_valid_test.pkl", 'rb') as f:
        ui_train, ui_valid, ui_test = pickle.load(f)
    emb = np.load("/home2/zh/data/twitter/emb.npy")
    return emb, ui_train, ui_valid, ui_test

def run0(model, data, user, phase, spt_qry_split=None, tot_user=296079):
    emb, ui_u = data
    if phase == 'train':
        itrue = ui_u['T']
        ui_u['F'] = np.random.choice(np.setdiff1d(np.arange(tot_user), np.concatenate((itrue, [user]))), len(itrue), False)
    onehot_i = torch.rand(2*len(ui_u['T']), 0).long().cuda()
    onehot_x = torch.rand(2*len(ui_u['T']), 0).double().cuda()
    multihot_list = []
    ctns_i = torch.tensor(np.concatenate((emb[ui_u['T']], emb[ui_u['F']]))).cuda().double()
    ctns_u = torch.tensor(emb[user]).repeat(ctns_i.shape[0], 1).cuda().double()
    ctns = torch.cat((ctns_u, ctns_i), 1)
    task_data = (onehot_i, onehot_x, multihot_list, ctns)
    label = torch.cat((torch.ones(ui_u['T'].shape), torch.zeros(ui_u['F'].shape))).cuda().double()
    spt_qry_perm = torch.randperm(len(label))
    if phase == 'train':
        if spt_qry_split == 'max(1/8, 4)':
            sqsplit = max(4, int(len(label)/8))
        else:
            raise Exception('Undeifined spt_qry_split')
        spt_idx = spt_qry_perm[:sqsplit]
        qry_idx = spt_qry_perm[-sqsplit:]
        # if (1-label[spt_idx]).sum() <= 0: return None
        # if (1-label[qry_idx]).sum() <= 0: return None
    else:
        spt_idx = spt_qry_perm[:0]
        qry_idx = spt_qry_perm[:]
        model.eval()
    pred, loss = model(task_data, label, spt_idx, qry_idx, phase=phase)
    if phase != 'train':
        model.train()
    return pred, loss, label[qry_idx]

if __name__ == '__main__':
    torch.cuda.set_device(2)
    emb, ui_train, ui_valid, ui_test = get_data()
    config = {'num_embeddings': 768*2,
            'embedding_dim': 32,
            'dim': [32*0+768*2, 64, 1],
            'dropout': [0, 0],
            'embedding_dim_meta': 32,
            'userl2': 32*0+768,
            'cluster_d': 32,
            'clusternum': [1, 3, 2, 1],
            'user_part': ([], [], list(range(768))),
            'inner_steps': 3,
            'batchsize': 256,
            'learning_rate': [0.5, 0.5],
            'update_lr': [0.05, (0, 0)],
            'seed': 81192}
    config['spt_qry_split'] = 'max(1/8, 4)'
    config['batchnum'] = int(75000*32/config['batchsize'])
    # config['eval_every'] = int(config['batchnum']/400)
    config['eval_every'] = 1
    setup_seed(config['seed'])
    model = MetaLearner(config).cuda().double()
    # user_set = np.array(list(set(ui[:,0])))
    # usernum = len(user_set)
    # perm = np.random.permutation(usernum)
    # train_usernum = int(0.8*usernum)
    # valid_usernum = int(0.9*usernum)
    optimizer = optim.SGD([{'params': model.net.parameters(), 'lr': config['learning_rate'][0]},
                            {'params': list(model.parameters())[len(list(model.net.parameters())):], 'lr': config['learning_rate'][1]}])
    traucs, vaaucs, trloss, valoss = [], [], [], []
    labels_chkp, preds_chkp, losses_chkp = [], [], []
    maxauc = 0.7
    test_auc, test_loss = 0, 0
    bestmodeldic = None
    for _ in tqdm(range(config['batchnum'])):
        preds, losses, labels = [], [], []
        for T_T in range(config['batchsize']):
            user = np.random.choice(list(ui_train.keys()), 1).item()
            data = (emb, ui_train[user])
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
            for user in ui_valid.keys():
                data = (emb, ui_valid[user])
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
                    for user in ui_test.keys():
                        data = (emb, ui_test[user])
                        pred, loss, label = run0(model, data, user, 'test')
                        preds.append(pred)
                        losses.append(loss)
                        labels.append(label)
                    test_auc = roc_auc_score(np.array(torch.cat(labels).detach().cpu()), np.array(torch.cat(preds).detach().cpu()))
                    test_loss = torch.stack(losses).mean().item()
            
            print('maxauc = {}, test_auc = {}'.format(maxauc, test_auc))
    
    pkl_file = 'res_twit.pkl'
    try:
        with open(pkl_file, 'rb') as f:
            res = pickle.load(f)
    except:
        res = []
    
    res_this_try = [config, traucs, trloss, vaaucs, valoss]

    if test_auc < 0.999:
        bestmodeldic = None
    res_this_try += [bestmodeldic, test_auc, test_loss]
    
    res.append(res_this_try)
    with open(pkl_file, 'wb') as f:
        pickle.dump(res, f)