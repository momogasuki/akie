from numpy import *
from sklearn.metrics import roc_auc_score

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn import ModuleList as ML
from torch.nn import ParameterList as PL
import numpy as np

import time

from autoint import AutoInt

def data():
    path = "/home2/zh/data/ml-100k/u.data"
    with open(path, 'r', encoding='ISO-8859-1') as f:
        lines = f.read().split('\n')
    interaction = array([[int(x) for x in line.split('\t')] for line in lines[:-1]])
    #print(interaction[:3])
    interaction[:,:2] -= 1
    #print(interaction[:3])
    interaction[interaction[:,2]<3, 2] = 0
    interaction[interaction[:,2]>3, 2] = 1
    interaction = interaction[interaction[:,2]<2]
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

class MetaLearner(nn.Module):
    def __init__(self, config):
        super(MetaLearner, self).__init__()
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
    
    def usingMetalearner():
        model = MetaLearner(config).cuda()
        perm = random.permutation(943)
        trains = perm[:800]
        tests = perm[800:]
        for _ in range(2000):
            losses = []
            for task_num in range(32):
                user = trains[random.randint(0,800)]
                tmp = interaction[interaction[:,0]==user]
                per = random.permutation(tmp.shape[0])
                spt = tmp[per[:5]]
                qry = tmp[per[-3:]]
                x_spt = torch.tensor(i_content[spt[:,1]]).float()
                y_spt = torch.tensor(spt[:,2]).float()
                x_qry = torch.tensor(i_content[qry[:,1]]).float()
                y_qry = torch.tensor(qry[:,2]).float()
                user_pf = torch.tensor(u_content[user]).float()
                res, loss = model(x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda(), user_pf.cuda())
                losses.append(loss)
            #print(torch.tensor(losses, requires_grad=True).mean())
            model.meta_optim.zero_grad()
            torch.stack(losses).mean().backward()
            model.meta_optim.step()
            if _ % 10 == 0:
                print(res, torch.stack(losses).mean())
    
    def onlyBaselearner():
        model = BaseLearner(config).cuda()
        perm = random.permutation(943)
        trains = perm[:800]
        tests = perm[800:]
        optimizer = optim.Adam(model.vars, lr=0.001)
        for _ in range(2000):
            losses, reses, labels = [], [], []
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
                reses.append(res.squeeze())
                labels.append(y_spt.squeeze())
            optimizer.zero_grad()
            torch.stack(losses).mean().backward()
            optimizer.step()
            if _ % 10 == 0:
                #print(res, torch.stack(losses).mean())
                #from IPython import embed; embed(); exit()
                print(roc_auc_score(array(torch.cat(labels).detach().cpu()), array(torch.cat(reses).detach().cpu())))
                print(torch.stack(losses).mean())
        torch.save(model.state_dict(), 'try.try')
        
        model = BaseLearner(config).cuda()
        model.load_state_dict(torch.load('try.try'))
        reses, labels = [], []
        for user in tests:
            tmp = interaction[interaction[:,0]==user]
            spt = tmp[:]
            x_spt = torch.tensor(i_content[spt[:,1]]).float()
            y_spt = torch.tensor(spt[:,2]).float()
            user_pf = torch.tensor(u_content[user]).float()
            res = model(user_pf.cuda(), x_spt.cuda(), model.vars)
            reses.append(res.squeeze())
            labels.append(y_spt.squeeze())
        print(roc_auc_score(array(torch.cat(labels).detach().cpu()), array(torch.cat(reses).detach().cpu())))       

    #usingMetalearner()
    onlyBaselearner()

if __name__ == '__main__':
    torch.cuda.set_device(5)
    setup_seed(81192)
    # x = 'onlybase'
    x = 'usemeta'
    if x == 'onlybase':
        # interaction, u_content, i_content = data()
        i_genre = np.load('/home2/zh/data/ml-1m/i_genre.npy')
        i_other = np.load('/home2/zh/data/ml-1m/i_other.npy')
        ui = np.load('/home2/zh/data/ml-1m/ui.npy')
        x_genre = np.load('/home2/zh/data/ml-1m/x_genre.npy')
        x_other = np.load('/home2/zh/data/ml-1m/x_other.npy')
        y = np.load('/home2/zh/data/ml-1m/y.npy')
        # main(interaction, u_content, i_content)
        config = {'num_embeddings': 3529,
                'embedding_dim': 16,
                'dim': [16*6+1, 64, 1]}
        model = BaseLearner(config).cuda().double()
        # from IPython import embed; embed(); exit()
        user_set = np.array(list(set(ui[:,0])))
        usernum = len(user_set)
        perm = random.permutation(usernum)
        train_usernum = int(0.8*usernum)
        valid_usernum = int(0.9*usernum)
        optimizer = optim.SGD(model.parameters(), lr=1)
        sample_cnt = 0
        idx0 = np.where(np.isin(ui[:,0], user_set[perm[0:train_usernum]]))[0]
        for _ in range(5000):
            losses, reses, labels = [], [], []
            for T_T in range(16):
                user = user_set[perm[random.randint(0,train_usernum)]]
                # idx = np.where(ui[:,0]==user)[0]
                # idx = idx[:-3] # keep 3 for query
                # idx = random.permutation(y.shape[0])[:100]
                idx = idx0[random.permutation(idx0.shape[0])[:400]]
                onehot_i = torch.tensor(i_other[idx,:-1]).cuda()
                onehot_x = torch.tensor(x_other[idx,:-1]).cuda()
                multihot_i = torch.tensor(i_genre[idx]).cuda()
                multihot_x = torch.tensor(x_genre[idx]).cuda()
                multihot_list = [(multihot_i, multihot_x)]
                ctns = torch.tensor(x_other[idx,-1:]).cuda()
                label = torch.tensor(y[idx]).cuda().double()
                res = model(onehot_i, onehot_x, multihot_list, ctns)
                loss = model.loss_func(res, label)
                losses.append(loss)
                reses.append(res)
                labels.append(label)
                sample_cnt += len(label)
            optimizer.zero_grad()
            torch.stack(losses).mean().backward()
            optimizer.step()
            if _ % 10 == 0:
                print('---- batch {} ---- sample_cnt: {}'.format(_, sample_cnt))
                # sample_cnt = 0
                # from IPython import embed; embed(); exit()
                idx = np.where(np.isin(ui[:,0], user_set[perm[train_usernum:valid_usernum]]))[0]
                onehot_i = torch.tensor(i_other[idx,:-1]).cuda()
                onehot_x = torch.tensor(x_other[idx,:-1]).cuda()
                multihot_i = torch.tensor(i_genre[idx]).cuda()
                multihot_x = torch.tensor(x_genre[idx]).cuda()
                multihot_list = [(multihot_i, multihot_x)]
                ctns = torch.tensor(x_other[idx,-1:]).cuda()
                label = torch.tensor(y[idx]).cuda().double()
                # from IPython import embed; embed(); exit()
                model.eval()
                res = model(onehot_i, onehot_x, multihot_list, ctns)
                model.train()
                loss = model.loss_func(res, label)
                # print(roc_auc_score(array(torch.cat(labels).detach().cpu()), array(torch.cat(reses).detach().cpu())))
                # print(torch.stack(losses).mean())
                # print(torch.cat(labels).shape)
                print(roc_auc_score(array(label.detach().cpu()), array(res.detach().cpu())))
                print(loss.mean())
                print(label.shape)
                for pg in optimizer.param_groups:
                    if pg['lr'] > 0.01:
                        pg['lr'] *= 0.98
        torch.save(model.state_dict(), 'try.try')
    else:
        # interaction, u_content, i_content = data()
        i_genre = np.load('/home2/zh/data/ml-1m/i_genre.npy')
        i_other = np.load('/home2/zh/data/ml-1m/i_other.npy')
        ui = np.load('/home2/zh/data/ml-1m/ui.npy')
        x_genre = np.load('/home2/zh/data/ml-1m/x_genre.npy')
        x_other = np.load('/home2/zh/data/ml-1m/x_other.npy')
        y = np.load('/home2/zh/data/ml-1m/y.npy')
        # main(interaction, u_content, i_content)
        config = {'num_embeddings': 3529,
                'embedding_dim': 16,
                'dim': [16*6+1, 64, 1],
                'dropout': [0, 0],
                'embedding_dim_meta': 32,
                'userl2': 32*4+0,
                'cluster_d': 128,
                'clusternum': [1, 3, 2, 1],
                'user_part': ([0,1,2,3], [], []),
                'inner_steps': 2,
                #'batchnum': 6000,
                'batchsize': 256,
                'learning_rate': [0.5, 0.5],
                'update_lr': [0.002, (0, 0.002)]}
        config['spt_qry_split'] = 'max(1/8, 4)'
        config['batchnum'] = int(75000*32/config['batchsize'])
        config['eval_every'] = int(config['batchnum']/400)
        model = MetaLearner(config).cuda().double()
        # from IPython import embed; embed(); exit()
        user_set = np.array(list(set(ui[:,0])))
        usernum = len(user_set)
        perm = random.permutation(usernum)
        train_usernum = int(0.8*usernum)
        valid_usernum = int(0.9*usernum)
        optimizer = optim.SGD([{'params': model.net.parameters(), 'lr': config['learning_rate'][0]},
                               {'params': list(model.parameters())[len(list(model.net.parameters())):], 'lr': config['learning_rate'][1]}])
        traucs, vaaucs, trloss, valoss = [], [], [], []
        labels_chkp, reses_chkp, losses_chkp = [], [], []
        checkt = time.clock()
        btm, tm, atm, vt, clk = 0,0,0,0,0
        for _ in range(config['batchnum']):
            losses, reses, labels = [], [], []
            for T_T in range(config['batchsize']):
                user = user_set[perm[random.randint(0,train_usernum)]]
                idx = np.where(ui[:,0]==user)[0]
                if len(idx) <= 10: continue
                onehot_i = torch.tensor(i_other[idx,:-1]).cuda()
                onehot_x = torch.tensor(x_other[idx,:-1]).cuda()
                multihot_i = torch.tensor(i_genre[idx]).cuda()
                multihot_x = torch.tensor(x_genre[idx]).cuda()
                multihot_list = [(multihot_i, multihot_x)]
                ctns = torch.tensor(x_other[idx,-1:]).cuda()
                task_data = (onehot_i, onehot_x, multihot_list, ctns)
                label = torch.tensor(y[idx]).cuda().double()
                spt_qry_perm = torch.randperm(len(idx))
                sqsplit = int(len(idx)/8)
                if sqsplit < 4: sqsplit = 4
                spt_idx = spt_qry_perm[:sqsplit]
                qry_idx = spt_qry_perm[-sqsplit:]
                if (1-label[spt_idx]).sum() <= 0: continue
                if (1-label[qry_idx]).sum() <= 0: continue
                btm += time.clock() - checkt; checkt = time.clock()
                pred, loss = model(task_data, label, spt_idx, qry_idx)
                tm += time.clock() - checkt; checkt = time.clock()
                losses.append(loss)
                reses.append(pred)
                labels.append(label[qry_idx])
                labels_chkp += labels
                reses_chkp += reses
                losses_chkp += losses
                atm += time.clock() - checkt; checkt = time.clock()
            if len(losses) == 0: continue
            optimizer.zero_grad()
            torch.stack(losses).mean().backward()
            optimizer.step()
            atm += time.clock() - checkt; checkt = time.clock()
            if _ % config['eval_every'] == 0:
                # from IPython import embed; embed()
                # print(list(model.parameters())[32])
                print('---- batch {} ----'.format(_))
                #print(pred)
                train_auc = roc_auc_score(array(torch.cat(labels_chkp).detach().cpu()), array(torch.cat(reses_chkp).detach().cpu()))
                train_loss = torch.stack(losses_chkp).mean().item()
                print('train_auc = {}, train_loss = {}'.format(train_auc, train_loss))
                labels_chkp, reses_chkp, losses_chkp = [], [], []
                traucs.append(train_auc)
                trloss.append(train_loss)
                losses, reses, labels = [], [], []
                for user in user_set[perm[train_usernum:valid_usernum]]:
                # user = user_set[perm[random.randint(train_usernum, valid_usernum)]]
                    idx = np.where(ui[:,0]==user)[0]
                    onehot_i = torch.tensor(i_other[idx,:-1]).cuda()
                    onehot_x = torch.tensor(x_other[idx,:-1]).cuda()
                    multihot_i = torch.tensor(i_genre[idx]).cuda()
                    multihot_x = torch.tensor(x_genre[idx]).cuda()
                    multihot_list = [(multihot_i, multihot_x)]
                    ctns = torch.tensor(x_other[idx,-1:]).cuda()
                    task_data = (onehot_i, onehot_x, multihot_list, ctns)
                    label = torch.tensor(y[idx]).cuda().double()
                    spt_qry_perm = torch.randperm(len(idx))
                    spt_idx = spt_qry_perm[:0]
                    qry_idx = spt_qry_perm[:]
                    # from IPython import embed; embed(); exit()
                    pred, loss = model(task_data, label, spt_idx, qry_idx, phase='valid')
                    losses.append(loss)
                    reses.append(pred)
                    labels.append(label[qry_idx])
                    # print(roc_auc_score(array(torch.cat(labels).detach().cpu()), array(torch.cat(reses).detach().cpu())))
                    # print(torch.stack(losses).mean())
                    # print(torch.cat(labels).shape)
                valid_auc = roc_auc_score(array(torch.cat(labels).detach().cpu()), array(torch.cat(reses).detach().cpu()))
                valid_loss = torch.stack(losses).mean().item()
                print('valid_auc = {}, valid_loss = {}'.format(valid_auc, valid_loss))
                vaaucs.append(valid_auc)
                valoss.append(valid_loss)
                print(torch.cat(labels).shape)
                vt += time.clock() - checkt; checkt = time.clock()
                print('before model: {}, model: {}, after model: {}, valid: {}'.format(btm, tm, atm, vt))
        torch.save(model.state_dict(), 'try2.try')
        import pickle
        pkl_file = 'res.pkl'
        with open(pkl_file, 'rb') as f:
            res = pickle.load(f)
        with open(pkl_file, 'wb') as f:
            res.append((config, traucs, trloss, vaaucs, valoss))
            pickle.dump(res, f)