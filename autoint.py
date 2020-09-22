from numpy import *
from sklearn.metrics import roc_auc_score

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn import ModuleList as ML
from torch.nn import ParameterList as PL
import numpy as np

import time

from util import setup_seed

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

if __name__ == '__main__':
    torch.cuda.set_device(6)
    i_genre = np.load('/home2/zh/data/ml-1m/i_genre.npy')
    i_other = np.load('/home2/zh/data/ml-1m/i_other.npy')
    ui = np.load('/home2/zh/data/ml-1m/ui.npy')
    x_genre = np.load('/home2/zh/data/ml-1m/x_genre.npy')
    x_other = np.load('/home2/zh/data/ml-1m/x_other.npy')
    y = np.load('/home2/zh/data/ml-1m/y.npy')
    config = {'num_embeddings': 3529,
            'num_ctns': 1,
            'fieldnum': 7,
            'embedding_dim': 32,
            'headnum': 8,
            'attention_dim': 32,
            'seed': 81192}
    setup_seed(config['seed'])
    model = AutoInt(config).cuda().double()
    user_set = np.array(list(set(ui[:,0])))
    usernum = len(user_set)
    perm = random.permutation(usernum)
    train_usernum = int(0.8*usernum)
    valid_usernum = int(0.9*usernum)
    idx0 = np.where(np.isin(ui[:,0], user_set[perm[0:train_usernum]]))[0]
    idx = idx0[random.permutation(idx0.shape[0])[:3200]]
    onehot_i = torch.tensor(i_other[idx,:-1]).cuda()
    onehot_x = torch.tensor(x_other[idx,:-1]).cuda()
    multihot_i = torch.tensor(i_genre[idx]).cuda()
    multihot_x = torch.tensor(x_genre[idx]).cuda()
    multihot_list = [(multihot_i, multihot_x)]
    ctns = torch.tensor(x_other[idx,-1:]).cuda()
    # label = torch.tensor(y[idx]).cuda().double()
    res = model(onehot_i, onehot_x, multihot_list, ctns)