import torch

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import DEFAULT_EPS


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


def geoRiskListnetLoss1(y_predicted, y_baselines, y_true, alpha=2, return_strategy=2, ob=0, corr=0):
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


def geoRiskSpearmanLoss1(y_predicted, y_baselines, y_true, alpha=2, return_strategy=2, ob=1, corr=0):
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




def lambdaLossreg(y_pred, y1, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE,
               weighing_scheme="ndcgLoss1_scheme", k=None, sigma=1., mu=10.,
               reduction="sum", reduction_log="binary", wr=1, ws=0):
    """
    LambdaLoss framework for LTR losses implementations, introduced in "The LambdaLoss Framework for Ranking Metric Optimization".
    Contains implementations of different weighing schemes corresponding to e.g. LambdaRank or RankNet.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param weighing_scheme: a string corresponding to a name of one of the weighing schemes
    :param k: rank at which the loss is truncated
    :param sigma: score difference weight used in the sigmoid function
    :param mu: optional weight used in NDCGLoss2++ weighing scheme
    :param reduction: losses reduction method, could be either a sum or a mean
    :param reduction_log: logarithm variant used prior to masking and loss reduction, either binary or natural
    :return: loss value, a torch.Tensor
    """
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

    if weighing_scheme != "ndcgLoss1_scheme":
        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

    ndcg_at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
    ndcg_at_k_mask[:k, :k] = 1

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.)
    y_true_sorted.clamp_(min=0.)

    # Here we find the gains, discounts and ideal DCGs per slate.
    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]
    maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

    # Here we apply appropriate weighing scheme - ndcgLoss1, ndcgLoss2, ndcgLoss2++ or no weights (=1.0)
    if weighing_scheme is None:
        weights = 1.
    else:
        weights = globals()[weighing_scheme](G, D, mu, true_sorted_by_preds)  # type: ignore

        # with torch.no_grad():
        #     mat = []
        #     mat.append(approxNDCGX(y_pred, y_true))
        #     for i in y1:
        #         mat.append(approxNDCGX(i, y_true))
        #     mat = torch.stack(mat).to(y_pred.device)
        #     device = y_pred.device
        #     mat = mat.t()
        #     selected_grisk = geoRisk(mat, 2.0, device, requires_grad=True)
        #     for q in range(weights.shape[0]):
        #         for d in range(weights.shape[1]):
        #             before = mat[q][0]
        #             mat[q][0] += torch.squeeze(weights[q][d])
        #             new_grisk = geoRisk(mat, 2.0, device, requires_grad=True)
        #             mat[q][0] = before
        #             weights[q][d] = new_grisk - selected_grisk

    # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
    scores_diffs[torch.isnan(scores_diffs)] = 0.
    weighted_probas = (torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
    if reduction_log == "natural":
        losses = torch.log(weighted_probas)
    elif reduction_log == "binary":
        losses = torch.log2(weighted_probas)
    else:
        raise ValueError("Reduction logarithm base can be either natural or binary")

    masked_losses = losses[padded_pairs_mask & ndcg_at_k_mask]
    if reduction == "sum":
        loss = -torch.sum(masked_losses)
    elif reduction == "mean":
        loss = -torch.mean(masked_losses)
    else:
        raise ValueError("Reduction method can be either sum or mean")

    if ws == 0:
        return loss
    elif ws == 1:
        return loss + wr * geoRiskListnetLoss1(y_pred, y1, y_true)
    elif ws == 2:
        return loss + wr * geoRiskSpearmanLoss1(y_pred, y1, y_true)
    return loss


def ndcgLoss1_scheme(G, D, *args):
    return (G / D)[:, :, None]


def ndcgLoss2_scheme(G, D, *args):
    pos_idxs = torch.arange(1, G.shape[1] + 1, device=G.device)
    delta_idxs = torch.abs(pos_idxs[:, None] - pos_idxs[None, :])
    deltas = torch.abs(torch.pow(torch.abs(D[0, delta_idxs - 1]), -1.) - torch.pow(torch.abs(D[0, delta_idxs]), -1.))
    deltas.diagonal().zero_()

    return deltas[None, :, :] * torch.abs(G[:, :, None] - G[:, None, :])


def lamdbaRank_scheme(G, D, *args):
    return torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(
        G[:, :, None] - G[:, None, :])


def ndcgLoss2PP_scheme(G, D, *args):
    return args[0] * ndcgLoss2_scheme(G, D) + lamdbaRank_scheme(G, D)


def rankNet_scheme(G, D, *args):
    return 1.


def rankNetWeightedByGTDiff_scheme(G, D, *args):
    return torch.abs(args[1][:, :, None] - args[1][:, None, :])


def rankNetWeightedByGTDiffPowed_scheme(G, D, *args):
    return torch.abs(torch.pow(args[1][:, :, None], 2) - torch.pow(args[1][:, None, :], 2))
