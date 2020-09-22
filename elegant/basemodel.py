from numpy import *
from sklearn.metrics import roc_auc_score

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn import ModuleList as ML
from torch.nn import ParameterList as PL
import numpy as np

import time

class AutoInt(nn.Module):
    def __init__(self, config):
        super(AutoInt, self).__init__()
        self.config = config
        self.embeddings = lambda idx, table: table.index_select(0, idx.reshape(-1)).reshape(*idx.shape, -1)

        # embedding table
        self.register_parameter(name='xx', param=nn.Parameter(torch.randn((config['num_embeddings'], config['embedding_dim']))))
        self.register_parameter(name='xy', param=nn.Parameter(torch.randn((config['num_ctns'], config['embedding_dim']))))
        self.att1 = ML([nn.Linear(in_features=config['embedding_dim'], out_features=config['headnum']*config['attention_dim'], bias=True),
                    nn.Linear(in_features=config['embedding_dim'], out_features=config['headnum']*config['attention_dim'], bias=True),
                    nn.Linear(in_features=config['embedding_dim'], out_features=config['headnum']*config['attention_dim'], bias=True),
                    nn.Linear(in_features=config['embedding_dim'], out_features=config['headnum']*config['attention_dim'], bias=True)])
        self.att2 = ML([nn.Linear(in_features=config['headnum']*config['attention_dim'], out_features=config['headnum']*config['attention_dim'], bias=True),
                    nn.Linear(in_features=config['headnum']*config['attention_dim'], out_features=config['headnum']*config['attention_dim'], bias=True),
                    nn.Linear(in_features=config['headnum']*config['attention_dim'], out_features=config['headnum']*config['attention_dim'], bias=True),
                    nn.Linear(in_features=config['headnum']*config['attention_dim'], out_features=config['headnum']*config['attention_dim'], bias=True)])
        self.logit = nn.Linear(in_features=config['fieldnum']*config['headnum']*config['attention_dim'], out_features=1, bias=True)

    def forward(self, onehot_i, onehot_x, multihot_list, ctns, vars=None):
        '''
        onehot_i: [spt_size, onehot_fieldnum], LongTensor
        onehot_x: [spt_size, onehot_fieldnum], maybe 0-1 FloatTensor
        multihot_list: list of (multihot_i, multihot_x) pairs
            multihot_i: [spt_size, maxhotnum], LongTensor
            multihot_x: [spt_size, maxhotnum], maybe 0-1 FloatTensor
        ctns: [spt_size, continuous_fieldnum], maybe FloatTensor
        '''
        if vars == None: vars = list(self.parameters())

        lookup_table = vars[0]
        lookup_table_ctns = vars[1]

        emb_dim = self.config['embedding_dim']
        att_dim = self.config['attention_dim']
        headnum = self.config['headnum']

        if 0 in onehot_i.shape:
            onehot_fields = []
        else:
            onehot_fields = self.embeddings(onehot_i, lookup_table) * onehot_x.repeat(emb_dim,1,1).permute(1,2,0)
            onehot_fields = [onehot_fields]
        multihot_fields = []
        for multihot_i, multihot_x in multihot_list:
            multihot_field = self.embeddings(multihot_i, lookup_table) * multihot_x.repeat(emb_dim,1,1).permute(1,2,0)
            multihot_fields.append(multihot_field.sum(dim=1).unsqueeze(1))
        
        # [N, m] * [m, d] = [N, m, d]
        ctns_fields = ctns.repeat(emb_dim,1,1).permute(1,2,0) * lookup_table_ctns
        ctns_fields = [ctns_fields]
        y_deep = torch.cat(onehot_fields+multihot_fields+ctns_fields, 1) # [spt_size, fieldnum, embedding_dim]
        # (dropout)

        # # attention_dim = block_shape
        # Q K V [embedding_dim, headnum*attention_dim]
        # Qy Ky Vy [spt_size, fieldnum, headnum*attention_dim]
        # Qy Ky Vy (split) --> [spt_size, fieldnum, headnum, attention_dim]
        # (concat) --> [spt_size*headnum, fieldnum, attention_dim]
        # (Qy Ky inner product) --> [spt_size*headnum, fieldnum, fieldnum]
        # (scale)
        # (softmax) --> [spt_size*headnum, fieldnum, fieldnum]
        # (dropout)
        # (*Vy) --> [spt_size*headnum, fieldnum, attention_dim]
        # (reshape) --> [spt_size, fieldnum, headnum*attention_dim]
        
        # (linear) --> [spt_size]

        QW, Qb, KW, Kb, VW, Vb, ResW, Resb = vars[2:10]
        Qy = F.linear(y_deep, QW, Qb)
        Ky = F.linear(y_deep, KW, Kb)
        Vy = F.linear(y_deep, VW, Vb)
        Res = F.linear(y_deep, ResW, Resb)

        Qy = torch.cat(Qy.split(att_dim, dim=2), dim=0)
        KyT = torch.cat(Ky.split(att_dim, dim=2), dim=0).transpose(1,2)
        Vy = torch.cat(Vy.split(att_dim, dim=2), dim=0)
        outputs = (Qy@KyT).softmax(2)@Vy
        outputs = torch.cat(outputs.split(outputs.shape[0]//headnum, dim=0), dim=2)

        outputs += Res

        outputs = F.relu(outputs)

        # 2
        QW, Qb, KW, Kb, VW, Vb, ResW, Resb = vars[10:18]
        Qy = F.linear(outputs, QW, Qb)
        Ky = F.linear(outputs, KW, Kb)
        Vy = F.linear(outputs, VW, Vb)
        Res = F.linear(outputs, ResW, Resb)

        Qy = torch.cat(Qy.split(att_dim, dim=2), dim=0)
        KyT = torch.cat(Ky.split(att_dim, dim=2), dim=0).transpose(1,2)
        Vy = torch.cat(Vy.split(att_dim, dim=2), dim=0)
        outputs = (Qy@KyT).softmax(2)@Vy
        outputs = torch.cat(outputs.split(outputs.shape[0]//headnum, dim=0), dim=2)

        outputs += Res

        outputs = F.relu(outputs)

        # flat
        outputs = outputs.reshape(outputs.shape[0], -1)

        logitW, logitb = vars[18:20]
        out = F.linear(outputs, logitW, logitb)

        # from IPython import embed; embed(); exit()
        return torch.sigmoid(out).squeeze()
    
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

class BaseLearner(nn.Module):
    def __init__(self, config):
        super(BaseLearner, self).__init__()
        # self.vars = nn.ParameterList()
        self.config = config
        # for (op, ed, out) in config['l1_u']:
        #     self.vars.append(nn.Parameter(torch.randn(ed-op, out)))
        # for (op, ed, out) in config['l1_i']:
        #     self.vars.append(nn.Parameter(torch.randn(ed-op, out)))
        # self.vars.append(nn.Parameter(torch.randn(*config['l2']))) # l2 = [yy, xx]
        # self.vars.append(nn.Parameter(torch.randn(config['l2'][0])))
        # self.vars.append(nn.Parameter(torch.randn(*config['l3']))) # l3 = [1, yy]
        # self.vars.append(nn.Parameter(torch.randn(config['l3'][0])))
        # self.embeddings_vars = nn.Embedding(config['num_embeddings'], config['embedding_dim'], padding_idx=0)
        self.lookup_table = nn.Parameter(torch.randn((config['num_embeddings'], config['embedding_dim'])))
        self.embeddings = lambda idx, table: table.index_select(0, idx.reshape(-1)).reshape(*idx.shape, -1)
        self.linear2 = nn.Linear(in_features=config['dim'][0], out_features=config['dim'][1], bias=True)
        self.linear3 = nn.Linear(in_features=config['dim'][1], out_features=config['dim'][2], bias=True)
        # self.linear4 = nn.Linear(in_features=config['l4'][0], out_features=config['l4'][1], bias=True)
        # self.w2 = nn.Parameter(torch.randn((config['l2'][1], config['l2'][0])))
        # self.b2 = nn.Parameter(torch.randn(config['l2'][1]))
        # self.w3 = nn.Parameter(torch.randn((config['l3'][1], config['l3'][0])))
        # self.b3 = nn.Parameter(torch.randn(config['l3'][1]))
        # print(list(self.parameters()))
    
    # def forward(self, xu, xi, vars=None):
    #     '''
    #     xu: [len(user)]
    #     xi: [spt_size, len(item)]
        
    #     return: [spt_size, 1]
    #     '''
    #     idx = 0
    #     to_concatu, to_concati = [], []
    #     for (op, ed, out) in self.config['l1_u']:
    #         to_concatu.append(xu[op:ed]@vars[idx])
    #         idx += 1
    #     for (op, ed, out) in self.config['l1_i']:
    #         to_concati.append(xi[:, op:ed]@vars[idx])
    #         idx += 1
    #     # [[out1], ..., [outm]], [[spt_size, out1], ..., [spt_size, outn]]
    #     xu = torch.cat(to_concatu, 0) # [outu]
    #     xi = torch.cat(to_concati, 1) # [spt_size, outi]
    #     x = torch.cat([xu.repeat(xi.shape[0],1), xi], dim=1) # [spt_size, outu+outi]
    #     #x = F.relu(x)
    #     w, b = vars[idx], vars[idx+1]
    #     x = F.linear(x, w, b)
    #     idx += 2
    #     x = torch.sigmoid(x)
    #     w, b = vars[idx], vars[idx+1]
    #     x = F.linear(x, w, b)
    #     idx += 2
    #     x = torch.sigmoid(x)
    #     return x

    def forward(self, onehot_i, onehot_x, multihot_list, ctns, vars=None):
        '''
        onehot_i: [spt_size, onehot_fieldnum], LongTensor
        onehot_x: [spt_size, onehot_fieldnum], maybe 0-1 FloatTensor
        multihot_list: list of (multihot_i, multihot_x) pairs
            multihot_i: [spt_size, maxhotnum], LongTensor
            multihot_x: [spt_size, maxhotnum], maybe 0-1 FloatTensor
        ctns: [spt_size, continuous_fieldnum], maybe FloatTensor
        '''
        if vars == None: vars = self.parameters()
        lookup_table, w2, b2, w3, b3= vars
        emb_dim = self.config['embedding_dim']
        if 0 in onehot_i.shape:
            onehot_fields = []
        else:
            onehot_fields = self.embeddings(onehot_i, lookup_table) * onehot_x.repeat(emb_dim,1,1).permute(1,2,0)
            onehot_fields = onehot_fields.reshape((onehot_fields.shape[0], -1))
            onehot_fields = [onehot_fields]
        multihot_fields = []
        for multihot_i, multihot_x in multihot_list:
            multihot_field = self.embeddings(multihot_i, lookup_table) * multihot_x.repeat(emb_dim,1,1).permute(1,2,0)
            multihot_fields.append(multihot_field.sum(dim=1))
        x = torch.cat(onehot_fields + multihot_fields + [ctns], dim=1)
        x = F.dropout(x, p=self.config['dropout'][0])
        x = F.linear(x, w2, b2)
        x = torch.relu(x)
        x = F.dropout(x, p=self.config['dropout'][1])
        x = F.linear(x, w3, b3)
        # x = torch.relu(x)
        # x = F.linear(x, w4, b4)
        # from IPython import embed; embed(); exit()
        return torch.sigmoid(x).squeeze()
    
    def loss_func(self, pred, label):
        pred = pred.squeeze()
        label = label.squeeze()
        num_1 = torch.sum(label)
        num_0 = torch.sum(1-label)
        num = num_1 + num_0
        # print(num_0.item(), num_1.item())
        # ret = -(pred.log()*label*num/(2*num_1+1e-10) + (1-pred).log()*(1-label)*num/(2*num_0+1e-10)).mean() # nn.BCELoss()
        ret = -(pred.log()*label + (1-pred).log()*(1-label)).mean() # nn.BCELoss()
        # print(ret.item())
        return ret