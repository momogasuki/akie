import torch
import torch.nn.functional as F
from torch import nn, optim

import numpy as np
from numpy import *

import math, pickle
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from akie import BaseLearner, setup_seed
from autoint import AutoInt

class WideDeep(nn.Module):
    def __init__(self, config):
        super(WideDeep, self).__init__()
        self.config = config
        self.wide_table = torch.randn((1, config['num_embeddings']))
        nn.init.kaiming_uniform_(self.wide_table, a=math.sqrt(5))
        self.wide_table = nn.Parameter(self.wide_table.reshape(-1,1))
        self.embed = lambda idx, table: table.index_select(0, idx.reshape(-1)).reshape(*idx.shape, -1) # usage: self.embed(idx, table)
        self.deep_table = nn.Parameter(torch.randn((config['num_embeddings'], config['embedding_dim'])))
        self.linear2 = nn.Linear(in_features=config['dim'][0], out_features=config['dim'][1], bias=True)
        self.linear3 = nn.Linear(in_features=config['dim'][1], out_features=config['dim'][2], bias=True)
        self.linear4 = nn.Linear(in_features=config['dim'][2], out_features=config['dim'][3], bias=True)
        self.lkrelu = nn.LeakyReLU(config['leaky'])
    
    def forward(self, onehot_i, onehot_x, multihot_list, ctns, vars=None):
        if vars == None: vars = self.parameters()
        wide_table, deep_table, w2, b2, w3, b3, w4, b4= vars
        # print(wide_table[:,:5])
        emb_dim = self.config['embedding_dim']
        onehot_fields = self.embed(onehot_i, wide_table) * onehot_x.repeat(emb_dim,1,1).permute(1,2,0)
        onehot_fields = onehot_fields.reshape((onehot_fields.shape[0], -1))
        multihot_fields = []
        for multihot_i, multihot_x in multihot_list:
            multihot_field = self.embed(multihot_i, wide_table) * multihot_x.repeat(emb_dim,1,1).permute(1,2,0)
            multihot_fields.append(multihot_field.sum(dim=1))
        wide_out = torch.cat([onehot_fields] + multihot_fields, dim=1).sum(dim=1)
        
        onehot_fields = self.embed(onehot_i, deep_table) * onehot_x.repeat(emb_dim,1,1).permute(1,2,0)
        onehot_fields = onehot_fields.reshape((onehot_fields.shape[0], -1))
        multihot_fields = []
        for multihot_i, multihot_x in multihot_list:
            multihot_field = self.embed(multihot_i, deep_table) * multihot_x.repeat(emb_dim,1,1).permute(1,2,0)
            multihot_fields.append(multihot_field.sum(dim=1))
        x = torch.cat([onehot_fields] + multihot_fields + [ctns], dim=1)
        x = F.dropout(x, p=self.config['dropout'][0])
        x = F.linear(x, w2, b2)
        x = self.lkrelu(x)
        x = F.dropout(x, p=self.config['dropout'][1])
        x = F.linear(x, w3, b3)
        x = self.lkrelu(x)
        x = F.dropout(x, p=self.config['dropout'][2])
        x = F.linear(x, w4, b4).squeeze()
        return torch.sigmoid(x + wide_out)
        # return torch.sigmoid(wide_out)
    
    def loss_func(self, pred, label):
        pred = pred.squeeze()
        label = label.squeeze()
        # num_1 = torch.sum(label)
        # num_0 = torch.sum(1-label)
        # num = num_1 + num_0
        # print(num_0.item(), num_1.item())
        # ret = -(pred.log()*label*num/(2*num_1+1e-10) + (1-pred).log()*(1-label)*num/(2*num_0+1e-10)).mean() # nn.BCELoss()
        # print(ret.item())
        ret = -(pred.log()*label + (1-pred).log()*(1-label)).mean()
        return ret
    
def main(config):
    setup_seed(config['seed'])
    model = AutoInt(config).cuda().double()
    user_set = np.array(list(set(ui[:,0])))
    usernum = len(user_set)
    perm = random.permutation(usernum)
    train_usernum = int(0.8*usernum)
    valid_usernum = int(0.9*usernum)
    optimizer = optim.SGD([{'params': list(model.parameters())[0:2], 'lr': config['lr'][0]},
                            {'params': list(model.parameters())[2:], 'lr': config['lr'][1]}], lr=1)
    idx0 = np.where(np.isin(ui[:,0], user_set[perm[0:train_usernum]]))[0]
    vaaucs = []
    maxauc = 0
    bestmodeldic = None
    for _ in tqdm(range(5000)):
        for T_T in range(1):
            user = user_set[perm[random.randint(0,train_usernum)]] # no use
            # idx = np.where(ui[:,0]==user)[0]
            # idx = idx[:-3] # keep 3 for query
            # idx = random.permutation(y.shape[0])[:100]
            idx = idx0[random.permutation(idx0.shape[0])[:3200]]
            onehot_i = torch.tensor(i_other[idx,:-1]).cuda()
            onehot_x = torch.tensor(x_other[idx,:-1]).cuda()
            multihot_i = torch.tensor(i_genre[idx]).cuda()
            multihot_x = torch.tensor(x_genre[idx]).cuda()
            multihot_list = [(multihot_i, multihot_x)]
            ctns = torch.tensor(x_other[idx,-1:]).cuda()
            label = torch.tensor(y[idx]).cuda().double()
            res = model(onehot_i, onehot_x, multihot_list, ctns)
            loss = model.loss_func(res, label)
        optimizer.zero_grad()
        
        loss.mean().backward()
        optimizer.step()
        if _ % 10 == 0:
            idx1 = np.where(np.isin(ui[:,0], user_set[perm[train_usernum:valid_usernum]]))[0]
            labels, reses = [], []
            k = 4
            for i in range(k):
                l = len(idx1)
                idx = idx1[int(i*l/k):int((i+1)*l/k)]
                onehot_i = torch.tensor(i_other[idx,:-1]).cuda()
                onehot_x = torch.tensor(x_other[idx,:-1]).cuda()
                multihot_i = torch.tensor(i_genre[idx]).cuda()
                multihot_x = torch.tensor(x_genre[idx]).cuda()
                multihot_list = [(multihot_i, multihot_x)]
                ctns = torch.tensor(x_other[idx,-1:]).cuda()
                label = torch.tensor(y[idx]).cuda().double()
                model.eval()
                with torch.no_grad():
                    res = model(onehot_i, onehot_x, multihot_list, ctns)
                model.train()
                labels.append(label)
                reses.append(res)
            label = torch.cat(labels)
            res = torch.cat(reses)
            loss = model.loss_func(res, label)
            # print(roc_auc_score(array(torch.cat(labels).detach().cpu()), array(torch.cat(reses).detach().cpu())))
            # print(torch.stack(losses).mean())
            # print(torch.cat(labels).shape)
            auc = roc_auc_score(array(label.detach().cpu()), array(res.detach().cpu()))
            vaaucs.append(auc)
            if auc > maxauc:
                bestmodeldic = model.state_dict()
                maxauc = auc
            print('auc = {}, maxauc = {}'.format(auc, maxauc))
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
        idx2 = np.where(np.isin(ui[:,0], user_set[perm[valid_usernum:]]))[0]
        labels, reses = [], []
        for i in range(k):
            l = len(idx2)
            idx = idx2[int(i*l/k):int((i+1)*l/k)]
            onehot_i = torch.tensor(i_other[idx,:-1]).cuda()
            onehot_x = torch.tensor(x_other[idx,:-1]).cuda()
            multihot_i = torch.tensor(i_genre[idx]).cuda()
            multihot_x = torch.tensor(x_genre[idx]).cuda()
            multihot_list = [(multihot_i, multihot_x)]
            ctns = torch.tensor(x_other[idx,-1:]).cuda()
            label = torch.tensor(y[idx]).cuda().double()
            model.eval()
            with torch.no_grad():
                res = model(onehot_i, onehot_x, multihot_list, ctns)
            model.train()
            labels.append(label)
            reses.append(res)
        label = torch.cat(labels)
        res = torch.cat(reses)
        loss = model.loss_func(res, label).item()
        auc = roc_auc_score(array(label.detach().cpu()), array(res.detach().cpu()))
        logg_this_try += [auc, loss]
        print('best valic auc = {}, test auc = {}'.format(maxauc, auc))
    logg.append(logg_this_try)
    with open(pkl_file, 'wb') as f:
        pickle.dump(logg, f)

if __name__ == '__main__':
    torch.cuda.set_device(6)
    i_genre = np.load('/home2/zh/data/ml-1m/i_genre.npy')
    i_other = np.load('/home2/zh/data/ml-1m/i_other.npy')
    ui = np.load('/home2/zh/data/ml-1m/ui.npy')
    x_genre = np.load('/home2/zh/data/ml-1m/x_genre.npy')
    x_other = np.load('/home2/zh/data/ml-1m/x_other.npy')
    y = np.load('/home2/zh/data/ml-1m/y.npy')
    config = {'num_embeddings': 3529,
            'embedding_dim': 32,
            'dim': [32*6+1, 128, 64, 1],
            'lr': [0.001, 1],
            'dropout': [0, 0, 0.5],
            'decay': [0, 1],
            'leaky': 0,
            'pkl_file': 'res_wddp.pkl',
            'seed': 81192}
    # config = {'num_embeddings': 3529,
    #         'embedding_dim': 16,
    #         'dim': [16*6+1, 64, 1],
    #         'lr': [0.01, 1],
    #         'dropout': [0, 0],
    #         'decay': [0.01, 0.98],
    #         'seed': 81192,
    #         'pkl_file': 'res_base.pkl'}
    # for lr in [[0.001, 1], [0.005, 0.5]]:
    #     for dropout in [[0, 0, 0], [0.1, 0, 0], [0, 0, 0.5], [0.1, 0.1, 0.1]]:
    #         for decay in [[0, 1]]:
    #             config['lr'] = lr
    #             config['dropout'] = dropout
    #             config['decay'] = decay
    #             print(config)
    #             main(config)
    main({'num_embeddings': 3529,
            'num_ctns': 1,
            'fieldnum': 7,
            'embedding_dim': 32,
            'headnum': 8,
            'attention_dim': 64,
            'lr': [0.05, 0.05],
            'decay': [0, 1],
            'pkl_file': 'res_autoint.pkl',
            'seed': 81192})