import numpy as np
import torch
from torch import nn


    

class MetaLearner(nn.Module):
    def __init__(self, config):
        super(MetaLearner, self).__init__()
        self.net = BaseLearner(config)
        self.config = config
        self.update_lr = 0.001
        self.construct_model()
        self.meta_optim = optim.Adam([{'params': self.net.vars, 'lr': 0.001},
                                      {'params': list(self.parameters())[len(list(self.net.vars)):], 'lr': 0.001}], lr=0.001)
    
    def forward(self, x_spt, y_spt, x_qry, y_qry, user_pf):
        '''
        x_spt: [spt_size, len(item)]
        y_spt: [spt_size]
        user_pf: [len(user)]
        '''
        task_emb = self.gen_task_emb(user_pf)
        clustered = self.cluster(task_emb)
        adapted_weights = self.adapt(self.net.vars, task_emb, clustered)
        for inner_idx in range(3):
            weights = adapted_weights if inner_idx == 0 else fast_weights
            pred = self.net(user_pf, x_spt, weights)
            loss = self.net.loss_func(pred, y_spt)
            grad = torch.autograd.grad(loss, weights, create_graph=True)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, weights)))
        pred_outer = self.net(user_pf, x_qry, fast_weights)
        loss_outer = self.net.loss_func(pred_outer, y_qry)
        #print(pred_outer, y_qry, loss_outer)
        
        return pred_outer, loss_outer
        
    def construct_model(self):
        self.linears1 = ML()
        for (op, ed, out) in self.config['userl1']:
            self.linears1.append(nn.Linear(in_features=ed-op, out_features=out, bias=False))
        self.cluster_d = self.config['cluster_d']
        self.linear2 = nn.Linear(in_features=self.config['userl2'][0], out_features=self.cluster_d, bias=False)
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
        for var in self.net.vars:
            self.adapt_linear.append(nn.Linear(in_features=self.cluster_d*2, out_features=np.prod(var.size()), bias=True))
        
    def gen_task_emb(self, user_pf):
        to_concat = []
        idx = 0
        for (op, ed, out) in self.config['userl1']:
            to_concat.append(self.linears1[idx](user_pf[op:ed]))
            idx += 1
        return self.linear2(torch.cat(to_concat, 0))
    
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
        for var in self.net.vars:
            o = torch.sigmoid(self.adapt_linear[idx](cated)).reshape(var.size()) * var
            adapted.append(o)
            idx += 1
        return adapted
    
    # def parameters(self):
    #     a = nn.ParameterList()
    #     a.extend(self.linears1 + self.adapt_linear)
    #     a.append(self.linear2)
    #     for i in self.cluster_centers[1:] + self.cluster_linears[1:]:
    #         a.extend(i)
    #     return a