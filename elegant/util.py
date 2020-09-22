import argparse

from numpy import *

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn import ModuleList as ML
from torch.nn import ParameterList as PL
import numpy as np

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def str2list(v):
    v=v.split(',')
    v=[int(_.strip('[]')) for _ in v]

    return v

def str2list2(v):
    v=v.split(',')
    v=[float(_.strip('[]')) for _ in v]

    return v

def str2list3(v):
    t = [x.strip(' ,)') for x in v.strip('[]').split('(')]
    return [float(t[0])] + [(int(x.split(',')[0]), float(x.split(',')[1])) for x in t[1:]]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpunum', type=int, default=0)

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--basemodel', type=str)
    
    # parser.add_argument('--num_embeddings', type=int, help='num of one-hot & multi-hot, in ml1m it\'s 3529.')
    # parser.add_argument('--num_ctns', type=int, help='num of continuous fields')
    # parser.add_argument('--fieldnum', type=int, help='total fields')
    parser.add_argument('--embedding_dim', type=int, help='embedding dimension in baselearner')
    parser.add_argument('--headnum', type=int, help='used in autoint')
    parser.add_argument('--attention_dim', type=int, help='used in autoint')
    parser.add_argument('--embedding_dim_meta', type=int, help='embedding dimension in metalearner')
    # parser.add_argument('--userl2', type=int, help='=embedding_dim_meta*user\'s concrete fieldnum + user\'s continuous num')
    parser.add_argument('--cluster_d', type=int, help='cluster dimension')
    parser.add_argument('--clusternum', type=str2list, help='list like [1, x, y, 1]')
    # parser.add_argument('--user_part')
    # I suggest the data-wise arguments can be automatically given
    parser.add_argument('--inner_steps', type=int, help='num of MAML\'s inner loop')
    parser.add_argument('--batchsize', type=int)
    parser.add_argument('--learning_rate', type=str2list2, help='outer loop lr for different parts')
    parser.add_argument('--update_lr', type=str2list3, help='inner loop lr like [x, (idx, y), (idx, z), ...]')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--spt_qry_split', type=str, help='split policy')
    # batchnum, eval_every
    parser.add_argument('--pkl_file', type=str)
    
    # used in baseline
    parser.add_argument('--lr', type=str2list2)
    parser.add_argument('--decay', type=str2list2, default=[0, 1])
    

    args = parser.parse_args()

    if args.dataset == 'ml1m':
        args.num_embeddings = 3529
        args.num_ctns = 1
        args.fieldnum = 7
        args.user_part = ([0,1,2,3], [], [])
    
    if args.embedding_dim_meta:
        args.userl2 = args.embedding_dim_meta * (len(args.user_part[0])+len(args.user_part[1])) + len(args.user_part[2])
    
        args.batchnum = int(75000*32/args.batchsize)
        args.eval_every = int(args.batchnum/400)

    return args