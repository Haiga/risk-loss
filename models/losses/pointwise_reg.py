import torch

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import DEFAULT_EPS
from allrank.models.metrics import ndcg

import torch
import torch.nn.functional as F

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.fast_soft_sort.pytorch_ops import soft_rank
from allrank.models.losses import DEFAULT_EPS


def zRisk(mat, alpha, device, requires_grad=False, i=0):
    alpha_tensor = torch.tensor([alpha], requires_grad=requires_grad, dtype=torch.float, device=device)
    si = torch.sum(mat[:, i])

    tj = torch.sum(mat, dim=1)
    n = torch.sum(tj)

    xij_eij = mat[:, i] - si * (tj / n)
    subden = si * (tj / n)
    den = torch.sqrt(subden + 1e-10)
    u = (den == 0) * torch.tensor([9e10], dtype=torch.float, requires_grad=requires_grad, device=device)

    den = u + den
    div = xij_eij / den

    less0 = (mat[:, i] - si * (tj / n)) / (den) < 0
    less0 = alpha_tensor * less0

    z_risk = div * less0 + div
    z_risk = torch.sum(z_risk)

    return z_risk


def geoRisk(mat, alpha, device, requires_grad=False, i=0):
    mat = mat * (mat > 0)
    si = torch.sum(mat[:, i])
    z_risk = zRisk(mat, alpha, device, requires_grad=requires_grad, i=i)

    num_queries = mat.shape[0]
    value = z_risk / num_queries
    m = torch.distributions.normal.Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
    ncd = m.cdf(value)
    return torch.sqrt((si / num_queries) * ncd + DEFAULT_EPS)


def compute_rank_correlation(first_array, second_array, device, u=PADDED_Y_VALUE / 1000):
    def _rank_correlation_(att_map, att_gd, device):
        n = torch.tensor(att_map.shape[0])
        upper = torch.tensor([6.0], device=device) * torch.sum((att_gd - att_map).pow(2))
        down = n * (n.pow(2) - torch.tensor([1.0], device=device))
        return (torch.tensor([1.0], device=device) - (upper / down)).mean(dim=-1)

    att = first_array.clone()
    grad_att = second_array.clone()

    a1 = soft_rank(att.unsqueeze(0), regularization_strength=u)
    a2 = soft_rank(grad_att.unsqueeze(0), regularization_strength=u)

    correlation = _rank_correlation_(a1[0], a2[0], device)

    return correlation


def spearmanLoss(y_predicted, y1, y_true, u=0.01):
    device = y_predicted.device
    # p_y_true = torch.squeeze(F.softmax(y_true, dim=1))
    # p_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))
    p_y_true = torch.squeeze(y_true)
    p_y_predicted = torch.squeeze(y_predicted)
    m = []
    for i in range(p_y_predicted.shape[0]):
        m.append(compute_rank_correlation(p_y_true[i], p_y_predicted[i], device, u=u))
    m = torch.stack(m)
    return -torch.mean(m)


def callGeorisk(mat, alpha, device, ob=1, return_strategy=2):
    selected_grisk = geoRisk(mat, alpha, device, requires_grad=True)
    if ob > 0:
        for i in range(mat.shape[1] - 2):
            u = geoRisk(mat, alpha, device, requires_grad=True, i=1 + i)
            if ob == 1:
                if u > selected_grisk:
                    selected_grisk = u
            else:
                if u < selected_grisk:
                    selected_grisk = u

    if return_strategy == 1:
        return -selected_grisk
    elif return_strategy == 2:
        return geoRisk(mat, alpha, device, requires_grad=True, i=-1) - selected_grisk
    elif return_strategy == 3:
        return (geoRisk(mat, alpha, device, requires_grad=True, i=-1) - selected_grisk) ** 2

    return None


def doLPred(true_smax, pred):
    preds_smax = F.softmax(pred, dim=1)
    preds_smax = preds_smax + DEFAULT_EPS
    preds_log = torch.log(preds_smax)
    return true_smax * preds_log


def geoRiskListnetLoss3(y_predicted, y_baselines, y_true, alpha=2, return_strategy=2, ob=0, corr=0):
    device = y_predicted.device

    # p_y_true = torch.squeeze(y_true)
    p_y_true = torch.squeeze(F.softmax(y_true, dim=1))
    p_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))

    correlations = []
    for i in range(p_y_predicted.shape[0]):
        if corr == 1:
            correlations.append(compute_rank_correlation(p_y_true[i], p_y_predicted[i], device))
        else:
            correlations.append(torch.nn.CosineSimilarity(dim=0)(p_y_true[i], p_y_predicted[i]))
    mat = [torch.stack(correlations)]

    if isinstance(y_baselines, list):
        for i in y_baselines:
            # p_y_baselines = torch.squeeze(i)
            p_y_baselines = torch.squeeze(F.softmax(i, dim=1))
            correlations_i = []
            for j in range(p_y_baselines.shape[0]):
                if corr == 1:
                    correlations_i.append(compute_rank_correlation(p_y_true[j], p_y_baselines[j], device))
                else:
                    correlations_i.append(torch.nn.CosineSimilarity(dim=0)(p_y_true[j], p_y_baselines[j]))
            correlations_i = torch.stack(correlations_i)
            mat.append(correlations_i)
    else:
        for i in range(y_baselines.shape[2]):
            # p_y_baselines = torch.squeeze(i)
            p_y_baselines = torch.squeeze(F.softmax(y_baselines[:, :, i], dim=1))
            correlations_i = []
            for j in range(p_y_baselines.shape[0]):
                if corr == 1:
                    correlations_i.append(compute_rank_correlation(p_y_true[j], p_y_baselines[j], device))
                else:
                    correlations_i.append(torch.nn.CosineSimilarity(dim=0)(p_y_true[j], p_y_baselines[j]))
            correlations_i = torch.stack(correlations_i)
            mat.append(correlations_i)

    mat.append(torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_true))

    mat = torch.stack(mat).to(device)
    mat = mat.t()

    return callGeorisk(mat, alpha, device, ob=ob, return_strategy=return_strategy)




def geoRiskSpearmanLoss3(y_predicted, y_baselines, y_true, alpha=2, return_strategy=2, ob=1, corr=0):
    device = y_predicted.device

    p_y_true = torch.squeeze(y_true)
    p_y_predicted = torch.squeeze(y_predicted)

    correlations = []
    for i in range(p_y_predicted.shape[0]):
        if corr == 1:
            correlations.append(compute_rank_correlation(p_y_true[i], p_y_predicted[i], device))
        else:
            correlations.append(torch.nn.CosineSimilarity(dim=0)(p_y_true[i], p_y_predicted[i]))
    mat = [torch.stack(correlations)]

    if isinstance(y_baselines, list):
        for i in y_baselines:
            p_y_baselines = torch.squeeze(i)
            correlations_i = []
            for j in range(p_y_baselines.shape[0]):
                if corr == 1:
                    correlations_i.append(compute_rank_correlation(p_y_true[j], p_y_baselines[j], device))
                else:
                    correlations_i.append(torch.nn.CosineSimilarity(dim=0)(p_y_true[j], p_y_baselines[j]))
            correlations_i = torch.stack(correlations_i)
            mat.append(correlations_i)
    else:
        for i in range(y_baselines.shape[2]):
            p_y_baselines = torch.squeeze(y_baselines[:, :, i])
            correlations_i = []
            for j in range(p_y_baselines.shape[0]):
                if corr == 1:
                    correlations_i.append(compute_rank_correlation(p_y_true[j], p_y_baselines[j], device))
                else:
                    correlations_i.append(torch.nn.CosineSimilarity(dim=0)(p_y_true[j], p_y_baselines[j]))
            correlations_i = torch.stack(correlations_i)
            mat.append(correlations_i)

    mat.append(torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_true))

    mat = torch.stack(mat).to(device)
    mat = mat.t()

    return callGeorisk(mat, alpha, device, ob=ob, return_strategy=return_strategy)

def pointwise_rmsereg(y_pred, y1, y_true, no_of_levels=5, normalized=1, padded_value_indicator=PADDED_Y_VALUE, wr=1, ws=0):
    """
    Pointwise RMSE loss.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param no_of_levels: number of unique ground truth values
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """


    y_pred = y_pred.clone()
    y_true = y_true.clone()

    if normalized:
        y_true = no_of_levels * y_true

    mask = y_true == padded_value_indicator
    valid_mask = (y_true != padded_value_indicator).type(torch.float32)

    y_true[mask] = 0
    y_pred[mask] = 0

    errors = (y_true - no_of_levels * y_pred)

    squared_errors = errors ** 2

    mean_squared_errors = torch.sum(squared_errors, dim=1) / torch.sum(valid_mask, dim=1)

    rmses = torch.sqrt(mean_squared_errors)

    # return torch.mean(rmses)
    loss = torch.mean(rmses)

    if ws == 0:
        return loss
    elif ws == 1:
        return loss + wr * geoRiskListnetLoss3(y_pred, y1, y_true)
    elif ws == 2:
        return loss + wr * geoRiskSpearmanLoss3(y_pred, y1, y_true)
    return loss