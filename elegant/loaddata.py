import numpy as np
import torch

class MyDataLoader():
    def __init__(self, args):
        self.args = args
        if args.dataset == 'ml1m':
            i_genre = np.load('/home2/zh/data/ml-1m/i_genre.npy')
            i_other = np.load('/home2/zh/data/ml-1m/i_other.npy')
            ui = np.load('/home2/zh/data/ml-1m/ui.npy')
            x_genre = np.load('/home2/zh/data/ml-1m/x_genre.npy')
            x_other = np.load('/home2/zh/data/ml-1m/x_other.npy')
            y = np.load('/home2/zh/data/ml-1m/y.npy')

            self.user_set = np.array(list(set(ui[:,0])))
            self.usernum = len(self.user_set)
            self.perm = np.random.permutation(self.usernum)
            self.train_usernum = int(0.8*self.usernum)
            self.valid_usernum = int(0.9*self.usernum)

            self.onehot_i = torch.tensor(i_other[:, :-1])
            self.onehot_x = torch.tensor(x_other[:, :-1])
            self.multihot_list = [(torch.tensor(i_genre), torch.tensor(x_genre))]
            self.ctns = torch.tensor(x_other[:, -1:])
            self.label = torch.tensor(y)
            self.ui = ui
            
            # all train_data idx
            self.idx0 = np.where(np.isin(ui[:,0], self.user_set[self.perm[0:self.train_usernum]]))[0]
            self.idx1 = np.where(np.isin(ui[:,0], self.user_set[self.perm[self.train_usernum:self.valid_usernum]]))[0]
            self.idx2 = np.where(np.isin(ui[:,0], self.user_set[self.perm[self.valid_usernum:]]))[0]
    
    def yield_data(self, phase):
        if phase == 'train':
            users = [self.user_set[self.perm[np.random.randint(0, self.train_usernum)]] for _ in range(self.args.batchsize)]
        elif phase == 'valid':
            users = self.user_set[self.perm[self.train_usernum:self.valid_usernum]]
        elif phase == 'test':
            users = self.user_set[self.perm[self.valid_usernum:]]
        else:
            raise Exception('Wrong code')

        for user in users:
            idx = np.where(self.ui[:,0]==user)[0]
            if phase == 'train':
                if len(idx) <= 10: continue
            task_data, label = self.databyidx(idx)
            spt_qry_perm = torch.randperm(len(idx))
            if phase == 'train':
                if self.args.spt_qry_split == 'max(1/8, 4)':
                    sqsplit = max(4, int(len(idx)/8))
                else:
                    raise Exception('Undeifined spt_qry_split')
                spt_idx = spt_qry_perm[:sqsplit]
                qry_idx = spt_qry_perm[-sqsplit:]
                # if (1-label[spt_idx]).sum() <= 0: continue
                # if (1-label[qry_idx]).sum() <= 0: continue
            else:
                spt_idx = spt_qry_perm[:0]
                qry_idx = spt_qry_perm[:]
            yield task_data, label, spt_idx, qry_idx
        
    def yield_data_baseline(self, phase):
        if phase == 'train':
            for _ in range(1):
                idx = self.idx0[np.random.permutation(self.idx0.shape[0])[:3200]]
                batch_data, label = self.databyidx(idx)
                yield batch_data, label
        else:
            idx_ = self.idx1 if phase == 'valid' else self.idx2
            k = 4
            for i in range(k):
                l = len(idx_)
                idx = idx_[int(i*l/k):int((i+1)*l/k)]
                batch_data, label = self.databyidx(idx)
                yield batch_data, label
                
    def databyidx(self, idx):
        onehot_i = self.onehot_i[idx].cuda()
        onehot_x = self.onehot_x[idx].cuda()
        multihot_list = [(a[idx].cuda(), b[idx].cuda()) for a,b in self.multihot_list]
        ctns = self.ctns[idx].cuda()
        label = self.label[idx].cuda().double()
        return (onehot_i, onehot_x, multihot_list, ctns), label