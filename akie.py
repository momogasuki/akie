from numpy import *

def data():
    path = "/home2/zh/data/ml-100k/u.data"
    with open(path, 'r', encoding='ISO-8859-1') as f:
        lines = f.read().split('\n')
    interaction = array([[int(x) for x in line.split('\t')] for line in lines[:-1]])
    #print(interaction[:3])
    interaction[:,:2] -= 1
    #print(interaction[:3])
    interaction[interaction[:,2]<=3, 2] = 0
    interaction[interaction[:,2]>3, 2] = 1
    #print(interaction[:3])

    def g(line):
        if line[2] == '':
            return [int(line[0])] + [0]*8 + line[-18:]
        return [int(line[0])] + list(eye(8)[int(line[2][-4:-1])-192]) + line[-18:]

    path = "/home2/zh/data/ml-100k/u.item"
    with open(path, 'r', encoding='ISO-8859-1') as f:
        lines = f.read().split('\n')
    #set([line.split('|')[2][-4:] for line in lines[:-1]])
    # line = lines[23].split('|')
    # array([int(line[0])] + list(eye(8)[int(line[2][-4:-1])-192]) + line[-18:]).astype(int)
    i_content = array([g(line.split('|')) for line in lines[:-1]]).astype(float)[:,1:]

    def h(line):
        #bl = [20, 30, 40, 50]
        a = [0]*28
        b = int(line[1])
        if b <= 20: a[0] = 1
        elif b <= 30: a[1] = 1
        elif b <= 40: a[2] = 1
        elif b <= 50: a[3] = 1
        else: a[4] = 1
        if line[2] == 'F': a[5] = 1
        else: a[6] = 1
        a[dict(zip(['administrator',
            'artist',
            'doctor',
            'educator',
            'engineer',
            'entertainment',
            'executive',
            'healthcare',
            'homemaker',
            'lawyer',
            'librarian',
            'marketing',
            'none',
            'other',
            'programmer',
            'retired',
            'salesman',
            'scientist',
            'student',
            'technician',
            'writer'], range(7,28)))[line[3]]] = 1
        return a

    path = "/home2/zh/data/ml-100k/u.user"
    with open(path, 'r', encoding='ISO-8859-1') as f:
        lines = f.read().split('\n')
    #set([line.split('|')[3] for line in lines[:-1]])
    u_content = array([h(line.split('|')) for line in lines[:-1]])
    #sum(i_content[:,0]==1)
    
    return interaction, u_content, i_content

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn import ModuleList as ML
from torch.nn import ParameterList as PL
import numpy as np

class BaseLearner(nn.Module):
    def __init__(self, config):
        super(BaseLearner, self).__init__()
        self.vars = nn.ParameterList()
        self.config = config
        for (op, ed, out) in config['l1_u']:
            self.vars.append(nn.Parameter(torch.randn(ed-op, out)))
        for (op, ed, out) in config['l1_i']:
            self.vars.append(nn.Parameter(torch.randn(ed-op, out)))
        self.vars.append(nn.Parameter(torch.randn(*config['l2']))) # l2 = [yy, xx]
        self.vars.append(nn.Parameter(torch.randn(config['l2'][0])))
        self.vars.append(nn.Parameter(torch.randn(*config['l3']))) # l3 = [1, yy]
        self.vars.append(nn.Parameter(torch.randn(config['l3'][0])))
    
    def forward(self, xu, xi, vars=None):
        '''
        xu: [len(user)]
        xi: [spt_size, len(item)]
        
        return: [spt_size, 1]
        '''
        idx = 0
        to_concatu, to_concati = [], []
        for (op, ed, out) in self.config['l1_u']:
            to_concatu.append(xu[op:ed]@vars[idx])
            idx += 1
        for (op, ed, out) in self.config['l1_i']:
            to_concati.append(xi[:, op:ed]@vars[idx])
            idx += 1
        # [[out1], ..., [outm]], [[spt_size, out1], ..., [spt_size, outn]]
        xu = torch.cat(to_concatu, 0) # [outu]
        xi = torch.cat(to_concati, 1) # [spt_size, outi]
        x = torch.cat([xu.repeat(xi.shape[0],1), xi], dim=1) # [spt_size, outu+outi]
        #x = F.relu(x)
        w, b = vars[idx], vars[idx+1]
        x = F.linear(x, w, b)
        idx += 2
        x = torch.sigmoid(x)
        w, b = vars[idx], vars[idx+1]
        x = F.linear(x, w, b)
        idx += 2
        x = torch.sigmoid(x)
        return x
    
    def loss_func(self, a, b):
        a = a.squeeze()
        b = b.squeeze()
        return -(a.log()*b + (1-a).log()*(1-b)).mean()

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
        
#         self.meta_optim.zero_grad()
#         loss_outer.backward()
#         self.meta_optim.step()
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
    
#     def parameters(self):
#         a = nn.ParameterList()
#         a.extend(self.linears1 + self.adapt_linear)
#         a.append(self.linear2)
#         for i in self.cluster_centers[1:] + self.cluster_linears[1:]:
#             a.extend(i)
#         return a

def main(interaction, u_content, i_content):
    torch.cuda.set_device(1)
    config = {'l1_u': [(0,5,3), (5,7,2), (7,28,10)],
            'l1_i': [(0,8,4), (8,26,10)],
            'l2': [20, 29],
            'l3': [1, 20],
            'userl1': [(0,5,2), (5,7,2), (7,28,8)],
            'userl2': [12, 1],
            'cluster_d': 5,
            'clusternum': [1, 3, 2, 1]}
    # model = MetaLearner(config).cuda()
    # perm = random.permutation(943)
    # trains = perm[:800]
    # tests = perm[800:]
    # for _ in range(2000):
    #     losses = []
    #     for task_num in range(32):
    #         user = trains[random.randint(0,800)]
    #         tmp = interaction[interaction[:,0]==user]
    #         per = random.permutation(tmp.shape[0])
    #         spt = tmp[per[:5]]
    #         qry = tmp[per[-3:]]
    #         x_spt = torch.tensor(i_content[spt[:,1]]).float()
    #         y_spt = torch.tensor(spt[:,2]).float()
    #         x_qry = torch.tensor(i_content[qry[:,1]]).float()
    #         y_qry = torch.tensor(qry[:,2]).float()
    #         user_pf = torch.tensor(u_content[user]).float()
    #         res, loss = model(x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda(), user_pf.cuda())
    #         losses.append(loss)
    #     #print(torch.tensor(losses, requires_grad=True).mean())
    #     model.meta_optim.zero_grad()
    #     torch.stack(losses).mean().backward()
    #     model.meta_optim.step()
    #     if _ % 10 == 0:
    #         print(res, torch.stack(losses).mean())
    model = BaseLearner(config).cuda()
    perm = random.permutation(943)
    trains = perm[:800]
    tests = perm[800:]
    optimizer = optim.Adam(model.vars, lr=0.001)
    for _ in range(2000):
        losses = []
        for task_num in range(32):
            user = trains[random.randint(0,800)]
            tmp = interaction[interaction[:,0]==user]
            per = random.permutation(tmp.shape[0])
            spt = tmp[per[:5]]
            x_spt = torch.tensor(i_content[spt[:,1]]).float()
            y_spt = torch.tensor(spt[:,2]).float()
            user_pf = torch.tensor(u_content[user]).float()
            res = model(user_pf.cuda(), x_spt.cuda(), model.vars)
            loss = model.loss_func(res, y_spt.cuda())
            losses.append(loss)
        optimizer.zero_grad()
        torch.stack(losses).mean().backward()
        optimizer.step()
        if _ % 10 == 0:
            print(res, torch.stack(losses).mean())

if __name__ == '__main__':
    interaction, u_content, i_content = data()
    main(interaction, u_content, i_content)