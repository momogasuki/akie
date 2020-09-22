import torch
import torch.nn.functional as F
from torch import nn, optim

import numpy as np
from numpy import *

import math, pickle
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from basemodel import AutoInt, WideDeep, BaseLearner
from util import parse_args, setup_seed
from loaddata import MyDataLoader

def model_eval(model, dataloader, phase):
    model.eval()
    labels, reses = [], []
    for batch_data, label in dataloader.yield_data_baseline(phase):
        with torch.no_grad():
            res = model(*batch_data)
        labels.append(label)
        reses.append(res)
    label = torch.cat(labels)
    res = torch.cat(reses)
    loss = model.loss_func(res, label).item()
    auc = roc_auc_score(array(label.detach().cpu()), array(res.detach().cpu()))
    return auc, loss

if __name__ == '__main__':
    args = parse_args()
    torch.cuda.set_device(args.gpunum)
    setup_seed(args.seed)
    dataloader = MyDataLoader(args)
    if args.basemodel == 'autoint':
        model = AutoInt(vars(args)).cuda().double()
    optimizer = optim.SGD([{'params': list(model.parameters())[0:2], 'lr': args.lr[0]},
                            {'params': list(model.parameters())[2:], 'lr': args.lr[1]}], lr=1)
    vaaucs = []
    maxauc = 0
    bestmodeldic = None
    for _ in tqdm(range(5000)):
        for batch_data, label in dataloader.yield_data_baseline('train'):
            res = model(*batch_data)
            loss = model.loss_func(res, label)
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        if _ % 10 == 0:
            auc, loss = model_eval(model, dataloader, 'valid')
            vaaucs.append(auc)
            if auc > maxauc:
                bestmodeldic = model.state_dict()
                maxauc = auc
            print('auc = {}, maxauc = {}'.format(auc, maxauc))
    
    try:
        with open(args.pkl_file, 'rb') as f:
            logg = pickle.load(f)
    except:
        logg = []
    logg_this_try = [args, vaaucs]
    if bestmodeldic:
        model.load_state_dict(bestmodeldic)
        auc, loss = model_eval(model, dataloader, 'test')
        logg_this_try += [auc, loss]
        print('best valic auc = {}, test auc = {}'.format(maxauc, auc))
    logg.append(logg_this_try)
    with open(args.pkl_file, 'wb') as f:
        pickle.dump(logg, f)