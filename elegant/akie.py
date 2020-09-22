from numpy import *
from sklearn.metrics import roc_auc_score

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn import ModuleList as ML
from torch.nn import ParameterList as PL
import numpy as np

from basemodel import AutoInt, WideDeep, BaseLearner

class MetaLearner(nn.Module):
    def __init__(self, args):
        super(MetaLearner, self).__init__()
        config = vars(args)
        if args.basemodel == 'autoint':
            self.net = AutoInt(config)
        self.config = config
        self.update_lr = [config['update_lr'][0]] * len(list(self.net.parameters()))
        for i, lr in config['update_lr'][1:]:
            self.update_lr[i] = lr
        self.construct_model()
        
    def construct_model(self):
        self.embeddings = nn.Embedding(self.config['num_embeddings'], self.config['embedding_dim_meta'], padding_idx=0)
        self.cluster_d = self.config['cluster_d']
        self.taskemb_linear =  nn.Linear(in_features=self.config['userl2'], out_features=self.cluster_d, bias=False)
        self.cluster_num = self.config['clusternum']
        self.cluster_centers = ML([PL([]),
                                PL([nn.Parameter(torch.randn(self.cluster_d)) for _ in range(self.cluster_num[1])]),
                                PL([nn.Parameter(torch.randn(self.cluster_d)) for _ in range(self.cluster_num[2])]),
                                PL([nn.Parameter(torch.randn(self.cluster_d)) for _ in range(1)])])
        self.cluster_linears = ML([ML([]),
                                ML([nn.Linear(in_features=self.cluster_d, out_features=self.cluster_d, bias=True)
                                     for _ in range(self.cluster_num[1])]),
                                ML([nn.Linear(in_features=self.cluster_d, out_features=self.cluster_d, bias=True)
                                     for _ in range(self.cluster_num[2])]),
                                ML([nn.Linear(in_features=self.cluster_d, out_features=self.cluster_d, bias=True)
                                     for _ in range(1)])])
        self.adapt_linear = ML()
        for var in self.net.parameters():
            self.adapt_linear.append(nn.Linear(in_features=self.cluster_d*2, out_features=np.prod(var.size()), bias=True))
    
    def forward(self, task_data, labels, spt_idx, qry_idx, phase='train'):
        '''
        task_data: (onehot_i, onehot_x, multihot_list, cnts), see BaseLearner
        labels: [spt_size+qry_size]
        '''
        task_emb = self.gen_task_emb(task_data)
        clustered = self.cluster(task_emb)
        fast_weights = self.adapt(self.net.parameters(), task_emb, clustered)
        # fast_weights = list(self.net.parameters())
        if phase == 'train':
            for inner_idx in range(self.config['inner_steps']):
                # weights = adapted_weights if inner_idx == 0 else fast_weights
                pred = self.net(*self.getdata(task_data, spt_idx), fast_weights)
                loss = self.net.loss_func(pred, labels[spt_idx])
                grad = torch.autograd.grad(loss, fast_weights, create_graph=True, allow_unused=True)
                grad = list(grad)
                for i in range(len(grad)):
                    if grad[i] is None: grad[i] = 0
                fast_weights = list(map(lambda p: p[1] - p[2] * p[0], zip(grad, fast_weights, self.update_lr)))
        pred_outer = self.net(*self.getdata(task_data, qry_idx), fast_weights)
        loss_outer = self.net.loss_func(pred_outer, labels[qry_idx])
        return pred_outer, loss_outer

    def getdata(self, data, idx):
        onehot_i, onehot_x, multihot_list, ctns = data
        return onehot_i[idx], onehot_x[idx], [(a[idx], b[idx]) for a,b in multihot_list], ctns[idx]
    
    def gen_task_emb(self, task_data):
        onehot_idx, multihot_idx, ctns_idx = [torch.tensor(x).long() for x in self.config['user_part']]
        onehot_i, onehot_x, multihot_list, ctns = task_data
        onehot_fields = self.embeddings(onehot_i[0, onehot_idx].long()) * onehot_x[0, onehot_idx].reshape((-1,1))
        onehot_fields = onehot_fields.reshape(-1)
        multihot_fields = []
        for idx in multihot_idx:
            multihot_i, multihot_x = multihot_list[idx]
            multihot_field = self.embeddings(multihot_i[0]) * multihot_x[0].reshape((-1,1))
            multihot_fields.append(multihot_field.sum(dim=0))
        x = torch.cat([onehot_fields] + multihot_fields + [ctns[0, ctns_idx]])
        x = self.taskemb_linear(x)
        # x = torch.relu(x)
        return x
    
    def cluster(self, task_emb):
        tmp = [task_emb]
        for layer in range(1, 4):
            to_softmax = []
            to_assign = []
            for idx in range(self.cluster_num[layer]):
                to_softmax.append(-torch.dist(self.cluster_centers[layer][idx], tmp[layer-1], 2))
                to_assign.append(torch.tanh(self.cluster_linears[layer][idx](tmp[layer-1])))
            softmaxed = F.softmax(torch.stack(to_softmax), dim=0)
            assigned = softmaxed@torch.stack(to_assign, dim=0)
            tmp.append(assigned) # dim to check
        return tmp[-1]

    def adapt(self, vars, task_emb, clustered):
        cated = torch.cat([task_emb, clustered])
        adapted = []
        idx = 0
        for var in self.net.parameters():
            o = torch.sigmoid(self.adapt_linear[idx](cated)).reshape(var.size()) * var
            adapted.append(o)
            idx += 1
        return adapted