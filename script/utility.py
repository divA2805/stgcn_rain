import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm
import torch
from sklearn.metrics import r2_score

def get_station_masks(vel_path):
    """
    Returns boolean masks for labeled and target stations.
    Labeled: columns with any nonzero value across all timesteps.
    Target: columns with all zero values (to be predicted).
    """
    vel = np.loadtxt(vel_path, delimiter=',')  # shape: (timesteps, stations)
    is_labeled = ~(np.all(vel == 0, axis=0))  # True for columns with at least one nonzero
    is_target = np.all(vel == 0, axis=0)      # True for columns that are all zeros
    return is_labeled, is_target

def calc_gso(dir_adj, gso_type):
    n_vertex = dir_adj.shape[0]

    if sp.issparse(dir_adj) == False:
        dir_adj = sp.csc_matrix(dir_adj)
    elif dir_adj.format != 'csc':
        dir_adj = dir_adj.tocsc()

    id = sp.identity(n_vertex, format='csc')

    # Symmetrizing an adjacency matrix
    adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)
    #adj = 0.5 * (dir_adj + dir_adj.transpose())
    
    if gso_type == 'sym_renorm_adj' or gso_type == 'rw_renorm_adj' \
        or gso_type == 'sym_renorm_lap' or gso_type == 'rw_renorm_lap':
        adj = adj + id
    
    if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' \
        or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
        row_sum = adj.sum(axis=1).A1
        row_sum_inv_sqrt = np.power(row_sum, -0.5)
        row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
        deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
        # A_{sym} = D^{-0.5} * A * D^{-0.5}
        sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

        if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
            sym_norm_lap = id - sym_norm_adj
            gso = sym_norm_lap
        else:
            gso = sym_norm_adj

    elif gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj' \
        or gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
        row_sum = np.sum(adj, axis=1).A1
        row_sum_inv = np.power(row_sum, -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        deg_inv = np.diag(row_sum_inv)
        # A_{rw} = D^{-1} * A
        rw_norm_adj = deg_inv.dot(adj)

        if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
            rw_norm_lap = id - rw_norm_adj
            gso = rw_norm_lap
        else:
            gso = rw_norm_adj

    else:
        raise ValueError(f'{gso_type} is not defined.')

    return gso

def calc_chebynet_gso(gso):
    if sp.issparse(gso) == False:
        gso = sp.csc_matrix(gso)
    elif gso.format != 'csc':
        gso = gso.tocsc()

    id = sp.identity(gso.shape[0], format='csc')
    # If you encounter a NotImplementedError, please update your scipy version to 1.10.1 or later.    
    eigvals = np.linalg.eigvalsh(gso.toarray())
    eigval_max = np.max(eigvals)

    # If the gso is symmetric or random walk normalized Laplacian,
    # then the maximum eigenvalue is smaller than or equals to 2.
    if eigval_max >= 2:
        gso = gso - id
    else:
        gso = 2 * gso / eigval_max - id

    return gso

def cnv_sparse_mat_to_coo_tensor(sp_mat, device):
    # convert a compressed sparse row (csr) or compressed sparse column (csc) matrix to a hybrid sparse coo tensor
    sp_coo_mat = sp_mat.tocoo()
    i = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)))
    v = torch.from_numpy(sp_coo_mat.data)
    s = torch.Size(sp_coo_mat.shape)

    if sp_mat.dtype == np.float32 or sp_mat.dtype == np.float64:
        return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.float32, device=device, requires_grad=False)
    else:
        raise TypeError(f'ERROR: The dtype of {sp_mat} is {sp_mat.dtype}, not been applied in implemented models.')

def evaluate_model(model, loss, data_iter, mask=None):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            out = model(x)
            y_pred = out[:, -1, ...]  # [batch, 1, n_nodes] or [batch, n_nodes]
            # Squeeze if necessary
            if y_pred.dim() == 3 and y_pred.shape[1] == 1:
                y_pred = y_pred.squeeze(1)  # [batch, n_nodes]
            if mask is not None:
                y_pred = y_pred[:, mask]
                y = y[:, mask]
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        mse = l_sum / n
        return mse

# def evaluate_metric(model, data_iter, scaler, mask=None):
#     model.eval()
#     with torch.no_grad():
#         mae, sum_y, mape, mse = [], [], [], []
#         all_y = []
#         all_y_pred = []
#         for x, y in data_iter:
#             if mask is not None:
#                 y = y[:, mask]
#             y_np = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
#             out = model(x)
#             y_pred = out[:, -1, ...]  # [batch, 1, n_nodes] or [batch, n_nodes]
#             # Squeeze if necessary
#             if y_pred.dim() == 3 and y_pred.shape[1] == 1:
#                 y_pred = y_pred.squeeze(1)  # [batch, n_nodes]
#             if mask is not None:
#                 y_pred = y_pred[:, mask]
#             y_pred_np = scaler.inverse_transform(y_pred.cpu().numpy()).reshape(-1)
#             d = np.abs(y_np - y_pred_np)
#             mae += d.tolist()
#             sum_y += y_np.tolist()
#             mape += (d / (y_np + 1e-8)).tolist()  # add small epsilon to avoid division by zero
#             mse += (d ** 2).tolist()
#             all_y.extend(y_np.tolist())
#             all_y_pred.extend(y_pred_np.tolist())
#         MAE = np.array(mae).mean()
#         #MAPE = np.array(mape).mean()
#         RMSE = np.sqrt(np.array(mse).mean())
#         #WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y))
#         WMAPE = np.sum(np.array(mae)) / (np.sum(np.array(sum_y)) + 1e-8)
#         # Calculate R2 Score
#         r2 = r2_score(all_y, all_y_pred)
#         return MAE, RMSE, WMAPE, r2
def evaluate_metric(model, data_iter, scaler, mask=None):
    model.eval()
    with torch.no_grad():
        mae, sum_y, mape, mse = [], [], [], []
        all_y = []
        all_y_pred = []
        for x, y in data_iter:
            y_np_full = scaler.inverse_transform(y.cpu().numpy())  # [batch, n_nodes]
            out = model(x)
            y_pred = out[:, -1, ...]
            if y_pred.dim() == 3 and y_pred.shape[1] == 1:
                y_pred = y_pred.squeeze(1)  # [batch, n_nodes]
            y_pred_np_full = scaler.inverse_transform(y_pred.cpu().numpy())
            if mask is not None:
                y_np = y_np_full[:, mask]
                y_pred_np = y_pred_np_full[:, mask]
            else:
                y_np = y_np_full
                y_pred_np = y_pred_np_full
            y_np = y_np.reshape(-1)
            y_pred_np = y_pred_np.reshape(-1)
            d = np.abs(y_np - y_pred_np)
            mae += d.tolist()
            sum_y += y_np.tolist()
            mape += (d / (y_np + 1e-8)).tolist()
            mse += (d ** 2).tolist()
            all_y.extend(y_np.tolist())
            all_y_pred.extend(y_pred_np.tolist())
        MAE = np.array(mae).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        WMAPE = np.sum(np.array(mae)) / (np.sum(np.array(sum_y)) + 1e-8)
        r2 = r2_score(all_y, all_y_pred)
        return MAE, RMSE, WMAPE, r2
        
