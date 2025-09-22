# import logging
# import os
# import gc
# import argparse
# import math
# import random
# import warnings
# import tqdm
# import numpy as np
# import pandas as pd
# from sklearn import preprocessing

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.utils as utils

# from script import dataloader, utility, earlystopping, opt
# from model import models
# from script.utility import get_station_masks


# #import nni

# def set_env(seed):
#     # Set available CUDA devices
#     # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
#     # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#     # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
#     # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     # torch.use_deterministic_algorithms(True)

# def get_parameters():
#     parser = argparse.ArgumentParser(description='STGCN')
#     parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
#     parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
#     parser.add_argument('--dataset', type=str, default='metr-la', choices=['metr-la', 'pems-bay', 'pemsd7-m'])
#     parser.add_argument('--n_his', type=int, default=3)
#     parser.add_argument('--n_pred', type=int, default=1, help='the number of time interval for predcition, default as 1')
#     parser.add_argument('--time_intvl', type=int, default=5)
#     parser.add_argument('--Kt', type=int, default=2)
#     parser.add_argument('--stblock_num', type=int, default=2)
#     parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
#     parser.add_argument('--Ks', type=int, default=3, choices=[3, 2])
#     parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv', choices=['cheb_graph_conv', 'graph_conv'])
#     parser.add_argument('--gso_type', type=str, default='sym_norm_lap', choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])
#     parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
#     parser.add_argument('--droprate', type=float, default=0.5)
#     parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
#     parser.add_argument('--weight_decay_rate', type=float, default=0.001, help='weight decay (L2 penalty)')
#     parser.add_argument('--batch_size', type=int, default=32)
#     parser.add_argument('--epochs', type=int, default=1000, help='epochs, default as 1000')
#     parser.add_argument('--opt', type=str, default='adamw', choices=['adamw', 'nadamw', 'lion'], help='optimizer, default as nadamw')
#     parser.add_argument('--step_size', type=int, default=10)
#     parser.add_argument('--gamma', type=float, default=0.95)
#     parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
#     args = parser.parse_args()
#     print('Training configs: {}'.format(args))

#     # For stable experiment results
#     set_env(args.seed)

#     # Running in Nvidia GPU (CUDA) or CPU
#     if args.enable_cuda and torch.cuda.is_available():
#         # Set available CUDA devices
#         # This option is crucial for multiple GPUs
#         # 'cuda' ≡ 'cuda:0'
#         device = torch.device('cuda')
#         torch.cuda.empty_cache() # Clean cache
#     else:
#         device = torch.device('cpu')
#         gc.collect() # Clean cache
    
#     Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num

#     # blocks: settings of channel size in st_conv_blocks and output layer,
#     # using the bottleneck design in st_conv_blocks
#     blocks = []
#     blocks.append([1])
#     for l in range(args.stblock_num):
#         blocks.append([64, 16, 64])
#     if Ko == 0:
#         blocks.append([128])
#     elif Ko > 0:
#         blocks.append([128, 128])
#     blocks.append([1])
    
#     return args, device, blocks

    
# def data_preparate(args, device):    
#     adj, n_vertex = dataloader.load_adj(args.dataset)
#     gso = utility.calc_gso(adj, args.gso_type)
#     if args.graph_conv_type == 'cheb_graph_conv':
#         gso = utility.calc_chebynet_gso(gso)
#     gso = gso.toarray()
#     gso = gso.astype(dtype=np.float32)
#     args.gso = torch.from_numpy(gso).to(device)

#     dataset_path = './data'
#     dataset_path = os.path.join(dataset_path, args.dataset)
#     data_col = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[0]
#     # recommended dataset split rate as train: val: test = 60: 20: 20, 70: 15: 15 or 80: 10: 10
#     # using dataset split rate as train: val: test = 70: 15: 15
#     val_and_test_rate = 0.20

#     len_val = int(math.floor(data_col * val_and_test_rate))
#     len_test = int(math.floor(data_col * val_and_test_rate))
#     len_train = int(data_col - len_val - len_test)
    
    
#     train, val, test = dataloader.load_data(args.dataset, len_train, len_val)
#     zscore = preprocessing.StandardScaler()
#     train = zscore.fit_transform(train)
#     val = zscore.transform(val)
#     test = zscore.transform(test)

#     x_train, y_train = dataloader.data_transform(train, args.n_his, args.n_pred, device)
#     x_val, y_val = dataloader.data_transform(val, args.n_his, args.n_pred, device)
#     x_test, y_test = dataloader.data_transform(test, args.n_his, args.n_pred, device)

#     train_data = utils.data.TensorDataset(x_train, y_train)
#     train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)
#     val_data = utils.data.TensorDataset(x_val, y_val)
#     val_iter = utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
#     test_data = utils.data.TensorDataset(x_test, y_test)
#     test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

#     return n_vertex, zscore, train_iter, val_iter, test_iter

# def prepare_model(args, blocks, n_vertex):
#     loss = nn.MSELoss()
#     es = earlystopping.EarlyStopping(delta=0.0, 
#                                      patience=args.patience, 
#                                      verbose=True, 
#                                      path="STGCN_" + args.dataset + ".pt")

#     if args.graph_conv_type == 'cheb_graph_conv':
#         model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(device)
#     else:
#         model = models.STGCNGraphConv(args, blocks, n_vertex).to(device)

#     if args.opt == "adamw":
#         optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
#     elif args.opt == "nadamw":
#         optimizer = optim.NAdam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, decoupled_weight_decay=True)
#     elif args.opt == "lion":
#         optimizer = opt.Lion(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
#     else:
#         raise ValueError(f'ERROR: The {args.opt} optimizer is undefined.')

#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

#     return loss, es, model, optimizer, scheduler

# def train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter, is_labeled):
#     for epoch in range(args.epochs):
#         l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
#         model.train()
#         for x, y in tqdm.tqdm(train_iter):
#             optimizer.zero_grad()
#             # out = model(x)
#             # print("Raw model output shape:", out.shape)
#             # #y_pred = model(x).view(len(x), -1)  # [batch_size, num_nodes]
#             # y_pred = model(x).reshape(len(x), -1)
#             # print('y_pred.shape:', y_pred.shape)
#             # print('y.shape:', y.shape)
#             # After getting output from your model
#             out = model(x)                   # e.g. [batch, n_pred, 1, n_nodes] or [batch, n_pred, n_nodes]
#             print("out.shape:", out.shape)
#             y_pred = out[:, -1, ...]         # Select last time step, shape: [batch, 1, n_nodes] or [batch, n_nodes]
#             print("y_pred.shape after slicing:", y_pred.shape)

#             # If y_pred has a singleton dimension, remove it
#             if y_pred.dim() == 3 and y_pred.shape[1] == 1:
#                 y_pred = y_pred.squeeze(1)   # Result: [batch, n_nodes]
#                 print("y_pred.shape after squeeze:", y_pred.shape)

#             y_pred_masked = y_pred[:, is_labeled]
#             y_masked = y[:, is_labeled]
#             print("y_pred_masked.shape:", y_pred_masked.shape)
#             print("y_masked.shape:", y_masked.shape)
#             l = loss(y_pred_masked, y_masked)
#             l.backward()
#             optimizer.step()
#             l_sum += l.item() * y.shape[0]
#             n += y.shape[0]
#         scheduler.step()
#         val_loss = val(model, val_iter, is_labeled, loss)
#         # GPU memory usage
#         gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
#         print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'.\
#             format(epoch+1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))

#         es(val_loss, model)
#         if es.early_stop:
#             print("Early stopping")
#             break

# @torch.no_grad()
# def val(model, val_iter, is_labeled, loss):
#     model.eval()

#     l_sum, n = 0.0, 0
#     for x, y in val_iter:
#         out = model(x)
#         print("val out.shape:", out.shape)
#         y_pred = out[:, -1, ...]
#         print("val y_pred.shape after slicing:", y_pred.shape)
#         if y_pred.dim() == 3 and y_pred.shape[1] == 1:
#             y_pred = y_pred.squeeze(1)
#             print("val y_pred.shape after squeeze:", y_pred.shape)
#         y_pred_masked = y_pred[:, is_labeled]
#         y_masked = y[:, is_labeled]
#         print("val y_pred_masked.shape:", y_pred_masked.shape)
#         print("val y_masked.shape:", y_masked.shape)
#         l = loss(y_pred_masked, y_masked)
#         l_sum += l.item() * y.shape[0]
#         n += y.shape[0]
#     return torch.tensor(l_sum / n)

# @torch.no_grad() 
# def test(zscore, loss, model, test_iter, args, is_target):
#     model.load_state_dict(torch.load("STGCN_" + args.dataset + ".pt"))
#     model.eval()

#     test_MSE = utility.evaluate_model(model, loss, test_iter, mask=is_target)
#     test_MAE, test_RMSE, test_WMAPE, test_R2 = utility.evaluate_metric(model, test_iter, zscore, mask=is_target)
#     print(f"Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f} | R2 {test_R2:.4f}")

# if __name__ == "__main__":
#     # Logging
#     #logger = logging.getLogger('stgcn')
#     #logging.basicConfig(filename='stgcn.log', level=logging.INFO)
#     logging.basicConfig(level=logging.INFO)

#     warnings.filterwarnings("ignore", category=FutureWarning)
#     warnings.filterwarnings("ignore", category=UserWarning)

#     args, device, blocks = get_parameters()
#     is_labeled, is_target = get_station_masks(f'./data/{args.dataset}/vel.csv')
#     n_vertex, zscore, train_iter, val_iter, test_iter = data_preparate(args, device)
#     loss, es, model, optimizer, scheduler = prepare_model(args, blocks, n_vertex)
#     train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter, is_labeled)
#     test(zscore, loss, model, test_iter, args, is_target)







import logging
import os
import gc
import argparse
import math
import random
import warnings
import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

from script import dataloader, utility, earlystopping, opt
from model import models
from script.utility import get_station_masks
# NEW: import for r2 calculation
from script.dataloader import load_rainfall_data
from script.utility import r2_for_targets

def set_env(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='metr-la', choices=['metr-la', 'pems-bay', 'pemsd7-m'])
    parser.add_argument('--n_his', type=int, default=3)
    parser.add_argument('--n_pred', type=int, default=1, help='the number of time interval for predcition, default as 1')
    parser.add_argument('--time_intvl', type=int, default=5)
    parser.add_argument('--Kt', type=int, default=2)
    parser.add_argument('--stblock_num', type=int, default=2)
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2])
    parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv', choices=['cheb_graph_conv', 'graph_conv'])
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap', choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.001, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1000, help='epochs, default as 1000')
    parser.add_argument('--opt', type=str, default='adamw', choices=['adamw', 'nadamw', 'lion'], help='optimizer, default as nadamw')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    set_env(args.seed)

    if args.enable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        gc.collect()
    
    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num

    blocks = []
    blocks.append([1])
    for l in range(args.stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])
    
    return args, device, blocks

# def data_preparate(args, device):    
#     adj, n_vertex = dataloader.load_adj(args.dataset)
#     gso = utility.calc_gso(adj, args.gso_type)
#     if args.graph_conv_type == 'cheb_graph_conv':
#         gso = utility.calc_chebynet_gso(gso)
#     gso = gso.toarray()
#     gso = gso.astype(dtype=np.float32)
#     args.gso = torch.from_numpy(gso).to(device)

#     dataset_path = './data'
#     dataset_path = os.path.join(dataset_path, args.dataset)
#     data_col = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[0]
#     # PATCH: train/val/test split ratio update
#     # 80:10:10 split
#     val_and_test_rate = 0.10
#     len_val = int(math.floor(data_col * val_and_test_rate))
#     len_test = int(math.floor(data_col * val_and_test_rate))
#     len_train = int(data_col - len_val - len_test)
    
#     #train, val, test = dataloader.load_data(args.dataset, len_train, len_val)
#     train, val, test = dataloader.load_data(args.dataset, len_train, len_val, args.n_his, args.n_pred)
#     zscore = preprocessing.StandardScaler()
#     train = zscore.fit_transform(train)
#     val = zscore.transform(val)
#     test = zscore.transform(test)

#     x_train, y_train = dataloader.data_transform(train, args.n_his, args.n_pred, device)
#     x_val, y_val = dataloader.data_transform(val, args.n_his, args.n_pred, device)
#     x_test, y_test = dataloader.data_transform(test, args.n_his, args.n_pred, device)

#     train_data = utils.data.TensorDataset(x_train, y_train)
#     train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)
#     val_data = utils.data.TensorDataset(x_val, y_val)
#     val_iter = utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
#     test_data = utils.data.TensorDataset(x_test, y_test)
#     test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

#     return n_vertex, zscore, train_iter, val_iter, test_iter


def data_preparate(args, device):    
    adj, n_vertex = dataloader.load_adj(args.dataset)
    gso = utility.calc_gso(adj, args.gso_type)
    if args.graph_conv_type == 'cheb_graph_conv':
        gso = utility.calc_chebynet_gso(gso)
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso).to(device)

    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, args.dataset)
    vel = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).values
    data_col = vel.shape[0]  # number of timesteps

    val_and_test_rate = 0.20
    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = data_col - len_val - len_test

    # Add a check here!
    min_len = args.n_his + args.n_pred
    if len_train < min_len or len_val < min_len or len_test < min_len:
        raise ValueError(f"Not enough timesteps in train/val/test splits! "
                         f"Got train={len_train}, val={len_val}, test={len_test}, min required={min_len}")

    train, val, test = dataloader.load_data(args.dataset, len_train, len_val)
    zscore = preprocessing.StandardScaler()
    train = zscore.fit_transform(train)
    val = zscore.transform(val)
    test = zscore.transform(test)

    x_train, y_train = dataloader.data_transform(train, args.n_his, args.n_pred, device)
    x_val, y_val = dataloader.data_transform(val, args.n_his, args.n_pred, device)
    x_test, y_test = dataloader.data_transform(test, args.n_his, args.n_pred, device)

    train_data = utils.data.TensorDataset(x_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)
    val_data = utils.data.TensorDataset(x_val, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    return n_vertex, zscore, train_iter, val_iter, test_iter
    
def prepare_model(args, blocks, n_vertex):
    loss = nn.MSELoss()
    es = earlystopping.EarlyStopping(delta=0.0, 
                                     patience=args.patience, 
                                     verbose=True, 
                                     path="STGCN_" + args.dataset + ".pt")

    if args.graph_conv_type == 'cheb_graph_conv':
        model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(device)
    else:
        model = models.STGCNGraphConv(args, blocks, n_vertex).to(device)

    if args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == "nadamw":
        optimizer = optim.NAdam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, decoupled_weight_decay=True)
    elif args.opt == "lion":
        optimizer = opt.Lion(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    else:
        raise ValueError(f'ERROR: The {args.opt} optimizer is undefined.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    return loss, es, model, optimizer, scheduler

def train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter, is_labeled):
    for epoch in range(args.epochs):
        l_sum, n = 0.0, 0
        model.train()
        for x, y in tqdm.tqdm(train_iter):
            optimizer.zero_grad()
            out = model(x)
            print("out.shape:", out.shape)
            y_pred = out[:, -1, ...]
            print("y_pred.shape after slicing:", y_pred.shape)
            if y_pred.dim() == 3 and y_pred.shape[1] == 1:
                y_pred = y_pred.squeeze(1)
                print("y_pred.shape after squeeze:", y_pred.shape)
            y_pred_masked = y_pred[:, is_labeled]
            y_masked = y[:, is_labeled]
            print("y_pred_masked.shape:", y_pred_masked.shape)
            print("y_masked.shape:", y_masked.shape)
            l = loss(y_pred_masked, y_masked)
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        scheduler.step()
        val_loss = val(model, val_iter, is_labeled, loss)
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))

        es(val_loss, model)
        if es.early_stop:
            print("Early stopping")
            break

@torch.no_grad()
def val(model, val_iter, is_labeled, loss):
    model.eval()
    l_sum, n = 0.0, 0
    for x, y in val_iter:
        out = model(x)
        print("val out.shape:", out.shape)
        y_pred = out[:, -1, ...]
        print("val y_pred.shape after slicing:", y_pred.shape)
        if y_pred.dim() == 3 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(1)
            print("val y_pred.shape after squeeze:", y_pred.shape)
        y_pred_masked = y_pred[:, is_labeled]
        y_masked = y[:, is_labeled]
        print("val y_pred_masked.shape:", y_pred_masked.shape)
        print("val y_masked.shape:", y_masked.shape)
        l = loss(y_pred_masked, y_masked)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return torch.tensor(l_sum / n)

# @torch.no_grad()
# def test(zscore, loss, model, test_iter, args, is_target):
#     model.load_state_dict(torch.load("STGCN_" + args.dataset + ".pt"))
#     model.eval()
#     test_MSE = utility.evaluate_model(model, loss, test_iter, mask=is_target)
#     test_MAE, test_RMSE, test_WMAPE, test_R2 = utility.evaluate_metric(model, test_iter, zscore, mask=is_target)
#     print(f"Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f} | R2 {test_R2:.4f}")

#     # NEW: get prediction array
#     preds = utility.get_predictions(model, test_iter, zscore)
#     # Load full rainfall data and features
#     features, rainfall, is_labeled_vec, is_target_vec = load_rainfall_data(f'./data/RainFallData_merged_1to30.csv')
#     print("Total stations:", len(is_labeled))
#     print("Labeled stations:", np.sum(is_labeled))
#     print("Target stations:", np.sum(is_target))
#     print("Example labeled indices:", np.where(is_labeled)[0][:10])
#     print("Example target indices:", np.where(is_target)[0][:10])
#     # print("Total stations:", len(is_target))
#     # print("Number of targets:", is_target.sum())
#     # print("Number of labeled:", is_labeled.sum())
#     # print("Target indices:", np.where(is_target)[0])
#     # print("Labeled indices:", np.where(is_labeled)[0])
#     print("preds shape:", preds.shape)
#     print("rainfall shape:", rainfall.shape)
#     print("features shape:", features.shape)
#     print("is_labeled shape:", is_labeled.shape)
#     print("is_target shape:", is_target.shape)
#     # Calculate R2 for each target station vs nearest labeled station
#     r2_scores = r2_for_targets(preds, rainfall, features, is_labeled_vec, is_target_vec)
#     print("R2 scores for all target stations:", r2_scores)

# Add this utility function near your other imports / utilities
def get_full_predictions(model, data, n_his, n_pred, device, scaler=None):
    """
    Reconstructs full predictions for all stations and timesteps from sliding windows.
    data: [timesteps, stations] (e.g., vel.csv values)
    Returns: preds_full [stations, timesteps]
    """
    n_vertex = data.shape[1]
    len_record = len(data)
    num = len_record - n_his - n_pred + 1
    x = np.zeros([num, 1, n_his, n_vertex])
    for i in range(num):
        x[i, :, :, :] = data[i:i+n_his].reshape(1, n_his, n_vertex)
    x_tensor = torch.Tensor(x).to(device)
    with torch.no_grad():
        out = model(x_tensor)
        y_pred = out[:, -1, ...]
        if y_pred.dim() == 3 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(1)
        preds = y_pred.cpu().numpy()
    # Inverse transform if scaler used
    if scaler is not None:
        preds = scaler.inverse_transform(preds)
    # Build full prediction array
    preds_full = np.zeros((n_vertex, len_record))
    for i in range(num):
        day = i + n_his + n_pred - 1
        preds_full[:, day] = preds[i]
    return preds_full

# Replace your current test() logic with the following:
@torch.no_grad() 
def test(zscore, loss, model, test_iter, args, is_target):
    # Load best weights
    model.load_state_dict(torch.load("STGCN_" + args.dataset + ".pt"))
    model.eval()

    # Standard metrics (unchanged)
    test_MSE = utility.evaluate_model(model, loss, test_iter, mask=is_target)
    test_MAE, test_RMSE, test_WMAPE, test_R2 = utility.evaluate_metric(model, test_iter, zscore, mask=is_target)
    print(f"Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f} | R2 {test_R2:.4f}")

    # NEW: get full predictions for all stations/timesteps
    import pandas as pd
    vel = pd.read_csv(f'./data/{args.dataset}/vel.csv').values
    preds_full = get_full_predictions(model, vel, args.n_his, args.n_pred, args.enable_cuda and torch.cuda.is_available() and torch.device('cuda') or torch.device('cpu'), scaler=zscore)
    print("preds_full shape:", preds_full.shape)

    # Align shapes for rainfall (if rainfall is shorter, crop predictions)
    from script.dataloader import load_rainfall_data
    features, rainfall, is_labeled_vec, is_target_vec = load_rainfall_data('./data/RainFallData_merged_1to30.csv')
    # min_days = min(rainfall.shape[1], preds_full.shape[1])
    # preds_for_rainfall = preds_full[:, :min_days]
    # print("Total stations:", len(is_labeled))
    # print("Labeled stations:", np.sum(is_labeled))
    # print("Target stations:", np.sum(is_target))
    # print("Example labeled indices:", np.where(is_labeled)[0][:10])
    # print("Example target indices:", np.where(is_target)[0][:10])
    # print("preds shape:", preds_full.shape)
    # print("rainfall shape:", rainfall.shape)
    # print("features shape:", features.shape)
    # print("is_labeled shape:", is_labeled.shape)
    # print("is_target shape:", is_target.shape)
    # #Calculate R2 for target stations vs nearest labeled station
    # from script.utility import r2_for_targets
    
    # r2_scores = r2_for_targets(preds_for_rainfall, rainfall, features, is_labeled_vec, is_target_vec)
    # print("R2 scores for all target stations:", r2_scores)
    rainfall_aligned = rainfall[:, -preds_full.shape[1]:]  # shape: (394, 29)
    preds_for_rainfall = preds_full[:, -rainfall_aligned.shape[1]:]  # shape: (394, 29)
    print("Aligned rainfall shape:", rainfall_aligned.shape)
    print("Aligned preds shape:", preds_for_rainfall.shape)
    print("Total stations:", len(is_labeled))
    print("Labeled stations:", np.sum(is_labeled))
    print("Target stations:", np.sum(is_target))
    print("Example labeled indices:", np.where(is_labeled)[0][:10])
    print("Example target indices:", np.where(is_target)[0][:10])
    print("features shape:", features.shape)
    print("is_labeled shape:", is_labeled.shape)
    print("is_target shape:", is_target.shape)

    print("Sample labeled rainfall:", rainfall_aligned[is_labeled][0])
    print("Sample predicted rainfall:", preds_for_rainfall[is_target][0])
    print("Is labeled rainfall constant?", np.all(rainfall_aligned[is_labeled][0] == rainfall_aligned[is_labeled][0][0]))
    r2_scores = r2_for_targets(preds_for_rainfall, rainfall_aligned, features, is_labeled_vec, is_target_vec)
    print("R2 scores for all target stations:", r2_scores)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    args, device, blocks = get_parameters()
    is_labeled, is_target = get_station_masks(f'./data/{args.dataset}/vel.csv')
    n_vertex, zscore, train_iter, val_iter, test_iter = data_preparate(args, device)
    loss, es, model, optimizer, scheduler = prepare_model(args, blocks, n_vertex)
    train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter, is_labeled)
    test(zscore, loss, model, test_iter, args, is_target)
