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


def geoRisk(mat, alpha, device, requires_grad=False, i=0, do_root=False):
    mat = mat * (mat > 0)
    si = torch.sum(mat[:, i])
    z_risk = zRisk(mat, alpha, device, requires_grad=requires_grad, i=i)

    num_queries = mat.shape[0]
    value = z_risk / num_queries
    m = torch.distributions.normal.Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
    ncd = m.cdf(value)
    if do_root:
        return torch.sqrt((si / num_queries) * ncd + DEFAULT_EPS)
    return (si / num_queries) * ncd


def compute_rank_correlation(first_array, second_array, device, u=0.0001):
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


def spearmanLoss(y_predicted, y1, y_true, u=0.0001):
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


def spearmanLossmulti(y_predicted, y1, y_true, u=0.01):
    soma = 0
    if isinstance(y1, list):

        for i in y1:
            soma += spearmanLoss(i, i, y_true, u=u)
            # i, y_true
    else:
        for i in range(y1.shape[2]):
            soma += spearmanLoss(y1[:, :, i], y1[:, :, i], y_true, u=u)

            # y1[:, :, i], y_true
    return soma


def callGeorisk(mat, alpha, device, ob=1, return_strategy=2, do_root=False):
    selected_grisk = geoRisk(mat, alpha, device, requires_grad=True)
    if ob > 0:
        for i in range(mat.shape[1] - 2):
            u = geoRisk(mat, alpha, device, requires_grad=True, i=1 + i, do_root=do_root)
            if ob == 1:
                if u > selected_grisk:
                    selected_grisk = u
            else:
                if u < selected_grisk:
                    selected_grisk = u

    if return_strategy == 1:
        return -selected_grisk
    elif return_strategy == 2:
        return geoRisk(mat, alpha, device, requires_grad=True, i=-1, do_root=do_root) - selected_grisk
    elif return_strategy == 3:
        return (geoRisk(mat, alpha, device, requires_grad=True, i=-1, do_root=do_root) - selected_grisk) ** 2

    return None


def doLPred(true_smax, pred):
    preds_smax = F.softmax(pred, dim=1)
    preds_smax = preds_smax + DEFAULT_EPS
    preds_log = torch.log(preds_smax)
    return true_smax * preds_log


def geoRiskListnetLoss(y_predicted, y_baselines, y_true, alpha=2, return_strategy=2, ob=0, corr=0):
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
        if (len(y_baselines.shape) > 2):
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


# def geoRiskListnetLoss(y_predicted, y_baselines, y_true, alpha=2, return_strategy=2, ob=1):
#     device = y_predicted.device
#
#     eps = DEFAULT_EPS
#     true_smax = F.softmax(y_true, dim=1)
#
#     preds_smax = F.softmax(y_predicted, dim=1)
#     preds_smax = preds_smax + eps
#     preds_log = torch.log(preds_smax)
#     c = torch.sum(true_smax * preds_log, dim=1)
#
#     mat = [c]
#
#     for i in y_baselines:
#         preds_smax = F.softmax(i, dim=1)
#         preds_smax = preds_smax + eps
#         preds_log = torch.log(preds_smax)
#         c = torch.sum(true_smax * preds_log, dim=1)
#         mat.append(c)
#
#     preds_smax = F.softmax(y_true, dim=1)
#     preds_smax = preds_smax + eps
#     preds_log = torch.log(preds_smax)
#     c = torch.sum(true_smax * preds_log, dim=1)
#     mat.append(c)
#
#     mat = torch.stack(mat).to(device)
#     mat = mat.t() - torch.min(mat)
#
#     return callGeorisk(mat, alpha, device, ob=ob, return_strategy=return_strategy)


def geoRiskSpearmanLoss(y_predicted, y_baselines, y_true, alpha=2, return_strategy=2, ob=0, corr=0, u=0.001):
    device = y_predicted.device

    p_y_true = torch.squeeze(y_true)
    p_y_predicted = torch.squeeze(y_predicted)

    correlations = []
    for i in range(p_y_predicted.shape[0]):
        if corr == 1:
            correlations.append(compute_rank_correlation(p_y_true[i], p_y_predicted[i], device, u=u))
        else:
            correlations.append(torch.nn.CosineSimilarity(dim=0)(p_y_true[i], p_y_predicted[i]))
    mat = [torch.stack(correlations)]

    if isinstance(y_baselines, list):
        for i in y_baselines:
            p_y_baselines = torch.squeeze(i)
            correlations_i = []
            for j in range(p_y_baselines.shape[0]):
                if corr == 1:
                    correlations_i.append(compute_rank_correlation(p_y_true[j], p_y_baselines[j], device, u=u))
                else:
                    correlations_i.append(torch.nn.CosineSimilarity(dim=0)(p_y_true[j], p_y_baselines[j]))
            correlations_i = torch.stack(correlations_i)
            mat.append(correlations_i)
    else:
        if (len(y_baselines.shape) > 2):
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


def approxNDCGLossGrisk(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE, alpha=1.):
    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    padded_mask = y_true == padded_value_indicator
    y_pred[padded_mask] = float("-inf")
    y_true[padded_mask] = float("-inf")

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)
    padded_pairs_mask.diagonal(dim1=-2, dim2=-1).zero_()

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.)
    y_true_sorted.clamp_(min=0.)

    # Here we find the gains, discounts and ideal DCGs per slate.
    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]
    maxDCGs = torch.sum((torch.pow(2, y_true_sorted) - 1) / D, dim=-1).clamp(min=eps)
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

    # Here we approximate the ranking positions according to Eqs 19-20 and later approximate NDCG (Eq 21)
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :])
    scores_diffs[~padded_pairs_mask] = 0.
    approx_pos = 1. + torch.sum(padded_pairs_mask.float() * (torch.sigmoid(-alpha * scores_diffs).clamp(min=eps)),
                                dim=-1)
    approx_D = torch.log2(1. + approx_pos)
    approx_NDCG = torch.sum((G / approx_D), dim=-1)

    return approx_NDCG


def approxNDCGLossGrisk2(y_pred, y1, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE, alpha=0.1):
    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    padded_mask = y_true == padded_value_indicator
    y_pred[padded_mask] = float("-inf")
    y_true[padded_mask] = float("-inf")

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)
    padded_pairs_mask.diagonal(dim1=-2, dim2=-1).zero_()

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.)
    y_true_sorted.clamp_(min=0.)

    # Here we find the gains, discounts and ideal DCGs per slate.
    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]
    maxDCGs = torch.sum((torch.pow(2, y_true_sorted) - 1) / D, dim=-1).clamp(min=eps)
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

    # Here we approximate the ranking positions according to Eqs 19-20 and later approximate NDCG (Eq 21)
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :])
    scores_diffs[~padded_pairs_mask] = 0.
    approx_pos = 1. + torch.sum(padded_pairs_mask.float() * (torch.sigmoid(-alpha * scores_diffs).clamp(min=eps)),
                                dim=-1)
    approx_D = torch.log2(1. + approx_pos)
    approx_NDCG = torch.sum((G / approx_D), dim=-1)

    return -torch.mean(approx_NDCG)


def geoRiskNDCG(y_predicted, y_baselines, y_true, alpha=2, return_strategy=2, ob=1, alphag=0.5):
    device = y_predicted.device

    correlations = approxNDCGLossGrisk(y_predicted, y_true, alpha=alphag)
    mat = [correlations]

    if isinstance(y_baselines, list):
        for i in y_baselines:
            correlations_i = approxNDCGLossGrisk(i, y_true, alpha=alphag)
            mat.append(correlations_i)
    else:
        temp = None
        # for i in range(y_baselines.shape[2]):
        #     p_y_baselines = torch.squeeze(y_baselines[:, :, i])
        #     correlations_i = []
        #     for j in range(p_y_baselines.shape[0]):
        #         if corr == 1:
        #             correlations_i.append(compute_rank_correlation(p_y_true[j], p_y_baselines[j], device))
        #         else:
        #             correlations_i.append(torch.nn.CosineSimilarity(dim=0)(p_y_true[j], p_y_baselines[j]))
        #     correlations_i = torch.stack(correlations_i)
        #     mat.append(correlations_i)

    mat.append(torch.nn.CosineSimilarity(dim=1)(y_true, y_true))

    mat = torch.stack(mat).to(device)
    mat = mat.t()

    return callGeorisk(mat, alpha, device, ob=ob, return_strategy=return_strategy)


def CorrScore(y_pred, y1, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE, ob=2):
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = float('-inf')

    preds_smax = F.softmax(y_pred, dim=1)
    true_smax = F.softmax(y_true, dim=1)

    preds_smax = preds_smax + eps
    true_smax = true_smax + eps
    preds_log = torch.log(preds_smax)
    trues_log = torch.log(true_smax)

    return torch.nn.CosineSimilarity(dim=1)(true_smax * preds_log, true_smax * trues_log)

def CorrScoreMean(y_pred, y1, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE, ob=2):
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = float('-inf')

    preds_smax = F.softmax(y_pred, dim=1)
    true_smax = F.softmax(y_true, dim=1)

    preds_smax = preds_smax + eps
    true_smax = true_smax + eps
    preds_log = torch.log(preds_smax)
    trues_log = torch.log(true_smax)

    return -torch.mean(torch.nn.CosineSimilarity(dim=1)(true_smax * preds_log, true_smax * trues_log))


def CorrScoremulti(y_pred, y1, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE, ob=2):
    soma = 0
    if isinstance(y1, list):

        for i in y1:
            soma += CorrScoreMean(i, i, y_true)
            # i, y_true
    else:
        for i in range(y1.shape[2]):
            soma += CorrScoreMean(y1[:, :, i], y1[:, :, i], y_true)

            # y1[:, :, i], y_true
    return soma


def geoRiskCorrScore(y_predicted, y1, y_true, alpha=2, return_strategy=2, ob=1, alphag=0.5):
    device = y_predicted.device
    mat = []

    if isinstance(y1, list):
        for i in y1:
            correlations_i = CorrScore(i, i, y_true)
            mat.append(correlations_i)
    else:
        for i in range(y1.shape[2]):
            correlations_i = CorrScore(y1[:, :, i], y1[:, :, i], y_true)
            mat.append(correlations_i)

    mat.append(torch.nn.CosineSimilarity(dim=1)(y_true, y_true))

    mat = torch.stack(mat).to(device)
    mat = mat.t()

    return callGeorisk(mat, alpha, device, ob=ob, return_strategy=return_strategy)


# torch.sum(torch.sum(losses, dim=1)* G, dim=1) - torch.min(torch.sum(torch.sum(losses, dim=1)* G, dim=1))
# torch.squeeze(weights)[:, 3] + torch.sum(torch.sum(losses, dim=1)* G, dim=1) - torch.min(torch.sum(torch.sum(losses, dim=1)* G, dim=1))
