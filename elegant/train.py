from akie import MetaLearner

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn import ModuleList as ML
from torch.nn import ParameterList as PL
import numpy as np

import time, pickle
from sklearn.metrics import roc_auc_score

from util import parse_args, setup_seed
from loaddata import MyDataLoader

def model_eval(model, dataloader, phase):
    model.eval()
    preds, losses, labels = [], [], []
    for task_data, label, spt_idx, qry_idx in dataloader.yield_data(phase):
        with torch.no_grad():
            pred, loss = model(task_data, label, spt_idx, qry_idx, phase)
        preds.append(pred)
        losses.append(loss)
        labels.append(label[qry_idx])
    auc = roc_auc_score(np.array(torch.cat(labels).detach().cpu()), np.array(torch.cat(preds).detach().cpu()))
    loss = torch.stack(losses).mean().item()
    model.train()
    return auc, loss

if __name__ == '__main__':
    args = parse_args()
    torch.cuda.set_device(args.gpunum)
    setup_seed(args.seed)
    dataloader = MyDataLoader(args)
    model = MetaLearner(args).cuda().double()
    optimizer = optim.SGD([{'params': model.net.parameters(), 'lr': args.learning_rate[0]},
                            {'params': list(model.parameters())[len(list(model.net.parameters())):], 'lr': args.learning_rate[1]}])
    traucs, vaaucs, trloss, valoss = [], [], [], []
    labels_chkp, preds_chkp, losses_chkp = [], [], []
    maxauc = 0
    test_auc, test_loss = 0, 0
    bestmodeldic = None
    for _ in range(args.batchnum):
        preds, losses, labels = [], [], []
        for task_data, label, spt_idx, qry_idx in dataloader.yield_data('train'):
            pred, loss = model(task_data, label, spt_idx, qry_idx, 'train')
            preds.append(pred)
            losses.append(loss)
            labels.append(label[qry_idx])
        preds_chkp += preds
        losses_chkp += losses
        labels_chkp += labels
        if losses:
            optimizer.zero_grad()
            torch.stack(losses).mean().backward()
            optimizer.step()

        if _ % args.eval_every == 0:
            print('---- batch {} ----'.format(_))
            train_auc = roc_auc_score(np.array(torch.cat(labels_chkp).detach().cpu()), np.array(torch.cat(preds_chkp).detach().cpu()))
            train_loss = torch.stack(losses_chkp).mean().item()
            print('train_auc = {}, train_loss = {}'.format(train_auc, train_loss))
            labels_chkp, preds_chkp, losses_chkp = [], [], []
            traucs.append(train_auc)
            trloss.append(train_loss)

            valid_auc, valid_loss = model_eval(model, dataloader, 'valid')
            print('valid_auc = {}, valid_loss = {}'.format(valid_auc, valid_loss))
            vaaucs.append(valid_auc)
            valoss.append(valid_loss)

            if valid_auc > maxauc:
                bestmodeldic = model.state_dict()
                maxauc = valid_auc

                if maxauc > 0.73:
                    test_auc, test_loss = model_eval(model, dataloader, 'test')
            
            print('maxauc = {}, test_auc = {}'.format(maxauc, test_auc))
    
    try:
        with open(args.pkl_file, 'rb') as f:
            res = pkl.load(f)
    except:
        res = []
    
    res_this_try = [args, traucs, trloss, vaaucs, valoss]

    if test_auc < 0.761:
        bestmodeldic = None
    res_this_try += [bestmodeldic, test_auc, test_loss]

    res.append(res_this_try)
    with open(args.pkl_file, 'wb') as f:
        pickle.dump(res, f)