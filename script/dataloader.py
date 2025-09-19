# import os
# import numpy as np
# import pandas as pd
# import scipy.sparse as sp
# import torch

# def load_adj(dataset_name):
#     dataset_path = './data'
#     dataset_path = os.path.join(dataset_path, dataset_name)
#     adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
#     adj = adj.tocsc()
    
#     if dataset_name == 'metr-la':
#         n_vertex = 394
#     elif dataset_name == 'pems-bay':
#         n_vertex = 325
#     elif dataset_name == 'pemsd7-m':
#         n_vertex = 228

#     return adj, n_vertex

# def load_data(dataset_name, len_train, len_val):
#     dataset_path = './data'
#     dataset_path = os.path.join(dataset_path, dataset_name)
#     vel = pd.read_csv(os.path.join(dataset_path, 'vel.csv'))

#     train = vel[: len_train]
#     val = vel[len_train: len_train + len_val]
#     test = vel[len_train + len_val:]
#     return train, val, test

# def data_transform(data, n_his, n_pred, device):
#     # produce data slices for x_data and y_data

#     n_vertex = data.shape[1]
#     len_record = len(data)
#     num = len_record - n_his - n_pred
#     print("len_record:", len_record)
#     print("n_his:", n_his)
#     print("n_pred:", n_pred)
#     print("num:", num)
#     #if num <= 0:
#        # raise ValueError(f"Not enough data: {len_record=} {n_his=} {n_pred=} {num=}")
#     x = np.zeros([num, 1, n_his, n_vertex])
#     y = np.zeros([num, n_vertex])
    
#     for i in range(num):
#         head = i
#         tail = i + n_his
#         x[i, :, :, :] = data[head: tail].reshape(1, n_his, n_vertex)
#         y[i] = data[tail + n_pred - 1]

#     return torch.Tensor(x).to(device), torch.Tensor(y).to(device)


# import pandas as pd
# import numpy as np

# def load_rainfall_data(csv_path):
#     df = pd.read_csv(csv_path)
#     features = df.iloc[:, 0:5].values  # [n_stations, 5]
#     rainfall = df.iloc[:, 5:35].values # [n_stations, 30]
#     is_labeled = np.any(rainfall != 0, axis=1)  # [n_stations]
#     is_target = ~is_labeled
#     return features, rainfall, is_labeled, is_target



import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

def load_adj(dataset_name):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
    adj = adj.tocsc()
    
    if dataset_name == 'metr-la':
        n_vertex = 394
    elif dataset_name == 'pems-bay':
        n_vertex = 325
    elif dataset_name == 'pemsd7-m':
        n_vertex = 228

    return adj, n_vertex

def load_data(dataset_name, len_train, len_val, n_his, n_pred):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    vel = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).values  # shape: [timesteps, stations]

    n_timesteps = vel.shape[0]
    # Ensure each split has enough for at least one sample in data_transform
    min_len = n_his + n_pred
    if len_train < min_len:
        raise ValueError(f"Train set too small for n_his={n_his}, n_pred={n_pred}")
    if len_val < min_len:
        # If not enough for val, skip val, add to train
        len_val = 0
    if n_timesteps - len_train - len_val < min_len:
        # If not enough for test, skip test, add to train
        len_test = 0
    else:
        len_test = n_timesteps - len_train - len_val

    train = vel[:len_train, :]
    val = vel[len_train:len_train + len_val, :] if len_val > 0 else np.empty((0, vel.shape[1]))
    test = vel[len_train + len_val:, :] if len_test > 0 else np.empty((0, vel.shape[1]))
    return train, val, test

def data_transform(data, n_his, n_pred, device):
    n_vertex = data.shape[1]
    len_record = len(data)
    num = len_record - n_his - n_pred + 1
    print("len_record:", len_record)
    print("n_his:", n_his)
    print("n_pred:", n_pred)
    print("num:", num)
    if num <= 0:
        raise ValueError(f"Not enough timesteps! Got len_record={len_record}, n_his={n_his}, n_pred={n_pred}, num={num}")
    x = np.zeros([num, 1, n_his, n_vertex])
    y = np.zeros([num, n_vertex])
    
    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data[head: tail].reshape(1, n_his, n_vertex)
        y[i] = data[tail + n_pred - 1]

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)

def load_rainfall_data(csv_path):
    df = pd.read_csv(csv_path)
    features = df.iloc[:, 0:5].values  # [n_stations, 5]
    rainfall = df.iloc[:, 5:35].values # [n_stations, 30]
    is_labeled = np.any(rainfall != 0, axis=1)  # [n_stations]
    is_target = ~is_labeled
    return features, rainfall, is_labeled, is_target
