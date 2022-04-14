import torch
import torch.nn.functional as F

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.fast_soft_sort.pytorch_ops import soft_rank, soft_sort
from allrank.models.losses import DEFAULT_EPS


def lambdaMask(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE, weighing_scheme=None, k=None,
               sigma=1., mu=10.,
               reduction="sum", reduction_log="binary", return_losses=False):
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

    if return_losses:
        return losses

    masked_losses = losses[padded_pairs_mask & ndcg_at_k_mask]

    return masked_losses


def zRisk(mat, alpha, device, requires_grad=False, i=0):
    alpha_tensor = torch.tensor([alpha], requires_grad=requires_grad, dtype=torch.float, device=device)
    si = torch.sum(mat[:, i])
    tj = torch.sum(mat, dim=1)
    n = torch.sum(tj)

    xij_eij = mat[:, i] - si * (tj / n)
    subden = si * (tj / n)
    den = torch.sqrt(subden + 1e-10)
    # den[torch.isnan(den)] = 0
    u = (den == 0) * torch.tensor([9e10], dtype=torch.float, requires_grad=requires_grad, device=device)
    den = u + den
    div = xij_eij / den

    less0 = (mat[:, i] - si * (tj / n)) / (den) < 0

    less0 = alpha_tensor * less0
    z_risk = div * less0 + div
    # z_risk[torch.isnan(z_risk)] = 0
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


def compute_rank_correlation(first_array, second_array, device, u=0.00001):
    def _rank_correlation_(att_map, att_gd, device):
        n = torch.tensor(att_map.shape[0])
        upper = torch.tensor([6.0], device=device) * torch.sum((att_gd - att_map).pow(2))
        down = n * (n.pow(2) - torch.tensor([1.0], device=device))
        return (torch.tensor([1.0], device=device) - (upper / down)).mean(dim=-1)

    att = first_array.clone()
    grad_att = second_array.clone()

    # u = 0.00001

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


def geoRiskSpearmanLoss(y_predicted, y1, y_true, alpha=4, return_strategy=1, listnet_transformation=4, negative=1,
                        add_ideal_ranking_to_mat=1, use_baseline=1):
    device = y_predicted.device

    p_y_true = torch.squeeze(y_true)
    p_y_predicted = torch.squeeze(y_predicted)
    p_y1 = torch.squeeze(y1)

    if listnet_transformation == 2:
        if use_baseline:
            correlations = torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_predicted)
            correlations2 = torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y1)
            mat = [correlations, correlations2, torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_true)]
        else:
            correlations = torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_predicted)
            mat = [correlations, torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_true)]
        mat = torch.stack(mat).to(device)
        mat = mat.t()

    elif listnet_transformation == 4:
        correlations = []
        for i in range(p_y_predicted.shape[0]):
            correlations.append(compute_rank_correlation(p_y_true[i], p_y_predicted[i], device))
        correlations = torch.stack(correlations)

        if use_baseline:
            correlations2 = []
            for i in range(p_y_predicted.shape[0]):
                correlations2.append(compute_rank_correlation(p_y_true[i], p_y1[i], device))
            correlations2 = torch.stack(correlations2)

            mat = [correlations, correlations2, torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_true)]
        else:
            mat = [correlations, torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_true)]
        mat = torch.stack(mat).to(device)
        mat = mat.t()

    factor = torch.tensor([negative], requires_grad=True, dtype=torch.float, device=device)
    if return_strategy == 1:
        return factor * geoRisk(mat, alpha, device, requires_grad=True)
    elif return_strategy == 2:
        return factor * (geoRisk(mat, alpha, device, requires_grad=True, i=-1) - geoRisk(mat, alpha, device,
                                                                                         requires_grad=True))
    elif return_strategy == 3:
        return factor * ((geoRisk(mat, alpha, device, requires_grad=True, i=-1) - geoRisk(mat, alpha, device,
                                                                                          requires_grad=True)) ** 2)

    return None


def geoRiskListnetLoss(y_predicted, y1, y_true, y_baselines=None, alpha=5, listnet_transformation=1, return_strategy=1,
                       negative=1, add_ideal_ranking_to_mat=1, use_baseline=1):
    device = y_predicted.device
    if not y_baselines is None:
        p_y_baselines = torch.squeeze(F.softmax(y_baselines, dim=1))

    p_y_true = torch.squeeze(F.softmax(y_true, dim=1))
    p_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))
    p_y1 = torch.squeeze(F.softmax(y1, dim=1))

    if listnet_transformation == 1:
        p_y_true_2 = p_y_true * p_y_true
        mat = [(p_y_true * p_y_predicted - p_y_true_2) ** 2]
    elif listnet_transformation == 2:
        if use_baseline:
            mat = [torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_predicted),
                   torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y1)]
        else:
            mat = [torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_predicted)]
    elif listnet_transformation == 3:
        p_y_true_2 = torch.sum(p_y_true * p_y_true, dim=1)
        mat = [(torch.sum(p_y_true * p_y_predicted, dim=1) - p_y_true_2) ** 2]
    elif listnet_transformation == 4:
        correlations = []
        for i in range(p_y_predicted.shape[0]):
            correlations.append(compute_rank_correlation(p_y_true[i], p_y_predicted[i], device))
        correlations = torch.stack(correlations)
        if use_baseline:
            correlations2 = []
            for i in range(p_y_predicted.shape[0]):
                correlations2.append(compute_rank_correlation(p_y_true[i], p_y1[i], device))
            correlations2 = torch.stack(correlations2)
            mat = [correlations, correlations2]
        else:
            mat = [correlations]

    if not y_baselines is None:
        for i in range(p_y_baselines.shape[2]):
            if listnet_transformation == 1:
                mat.append((p_y_true * p_y_baselines[:, :, i] - p_y_true_2) ** 2)
            elif listnet_transformation == 2:
                mat.append(torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_baselines[:, :, i]))
            elif listnet_transformation == 3:
                mat.append((torch.sum(p_y_true * p_y_baselines[:, :, i], dim=1) - p_y_true_2) ** 2)

    # if add_ideal_ranking_to_mat == 1:# do nothing
    if add_ideal_ranking_to_mat == 2:
        if listnet_transformation == 1:
            mat.append((p_y_true * p_y_true - p_y_true_2) ** 2)  # adicionar 0 diretamente?
        elif listnet_transformation == 2:
            mat.append(torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_true))  # adicionar 1 diretamente?
        elif listnet_transformation == 3:
            mat.append((torch.sum(p_y_true * p_y_true, dim=1) - p_y_true_2) ** 2)
        elif listnet_transformation == 4:
            mat.append(torch.nn.CosineSimilarity(dim=1)(p_y_true, p_y_true))  # adicionar 1 diretamente?

    mat = torch.stack(mat).to(device)
    if listnet_transformation == 1:
        mat = torch.sum(mat, dim=2)
    mat = mat.t()

    # listnet_transformation==1 or 3 deixa como melhor efetividade a maior distância, temos que inverter
    if listnet_transformation == 1 or listnet_transformation == 3:
        mat = -mat + torch.max(mat)

    factor = torch.tensor([negative], requires_grad=True, dtype=torch.float, device=device)
    if return_strategy == 1:
        return factor * geoRisk(mat, alpha, device, requires_grad=True)
    elif return_strategy == 2:
        return factor * (geoRisk(mat, alpha, device, requires_grad=True, i=-1) - geoRisk(mat, alpha, device,
                                                                                         requires_grad=True))
    elif return_strategy == 3:
        return factor * ((geoRisk(mat, alpha, device, requires_grad=True, i=-1) - geoRisk(mat, alpha, device,
                                                                                          requires_grad=True)) ** 2)

    return None


def geoRiskLambdaLoss(y_predicted, y1, y_true, y_baselines=None, alpha=5, listnet_transformation=1, return_strategy=1,
                      negative=1, add_ideal_ranking_to_mat=1, weighing_scheme="ndcgLoss2PP_scheme", use_baseline=1):
    device = y_predicted.device
    if not y_baselines is None:
        p_y_baselines = torch.squeeze(F.softmax(y_baselines, dim=1))

    p_y_true = torch.squeeze(F.softmax(y_true, dim=1)).to(device)
    p_y_predicted = torch.squeeze(F.softmax(y_predicted, dim=1))
    p_y1 = torch.squeeze(F.softmax(y1, dim=1))

    if listnet_transformation == 1:
        p_y_true_y_true = torch.sum(lambdaMask(p_y_true, p_y_true, weighing_scheme=weighing_scheme, return_losses=True),
                                    dim=1)
        p_y_predicted_y_true = torch.sum(
            lambdaMask(p_y_predicted, p_y_true, weighing_scheme=weighing_scheme, return_losses=True),
            dim=1)
        mat = [(p_y_predicted_y_true - p_y_true_y_true) ** 2]
    elif listnet_transformation == 2:
        p_y_true_y_true = torch.sum(lambdaMask(p_y_true, p_y_true, weighing_scheme=weighing_scheme, return_losses=True),
                                    dim=1)
        p_y_predicted_y_true = torch.sum(
            lambdaMask(p_y_predicted, p_y_true, weighing_scheme=weighing_scheme, return_losses=True),
            dim=1)
        p_y1_y_true = torch.sum(
            lambdaMask(p_y1, p_y_true, weighing_scheme=weighing_scheme, return_losses=True),
            dim=1)
        if use_baseline:
            mat = [torch.nn.CosineSimilarity(dim=1)(p_y_true_y_true, p_y_predicted_y_true),
                   torch.nn.CosineSimilarity(dim=1)(p_y_true_y_true, p_y1_y_true)]
        else:
            mat = [torch.nn.CosineSimilarity(dim=1)(p_y_true_y_true, p_y_predicted_y_true)]

    elif listnet_transformation == 4:
        p_y_true_y_true = torch.sum(lambdaMask(p_y_true, p_y_true, weighing_scheme=weighing_scheme, return_losses=True),
                                    dim=1)
        p_y_predicted_y_true = torch.sum(
            lambdaMask(p_y_predicted, p_y_true, weighing_scheme=weighing_scheme, return_losses=True),
            dim=1)

        p_y1_y_true = torch.sum(
            lambdaMask(p_y1, p_y_true, weighing_scheme=weighing_scheme, return_losses=True),
            dim=1)

        correlations = []
        for i in range(p_y_predicted_y_true.shape[0]):
            correlations.append(compute_rank_correlation(p_y_true_y_true[i], p_y_predicted_y_true[i], device))
        correlations = torch.stack(correlations)

        correlations2 = []
        for i in range(p_y_predicted_y_true.shape[0]):
            correlations2.append(compute_rank_correlation(p_y_true_y_true[i], p_y1_y_true[i], device))
        correlations2 = torch.stack(correlations2)

        if use_baseline:
            mat = [correlations, correlations2]
        else:
            mat = [correlations]

    if not y_baselines is None:
        for i in range(p_y_baselines.shape[2]):
            if listnet_transformation == 1:
                p_y_predicted_i = torch.sum(
                    lambdaMask(p_y_baselines[:, :, i], p_y_true, weighing_scheme=weighing_scheme, return_losses=True),
                    dim=1)
                mat.append((p_y_predicted_i - p_y_true_y_true) ** 2)
            elif listnet_transformation == 2:
                p_y_predicted_i = torch.sum(
                    lambdaMask(p_y_baselines[:, :, i], p_y_true, weighing_scheme=weighing_scheme, return_losses=True),
                    dim=1)
                mat.append(torch.nn.CosineSimilarity(dim=1)(p_y_true_y_true, p_y_predicted_i))

    # if add_ideal_ranking_to_mat == 1:# do nothing
    if add_ideal_ranking_to_mat == 2:
        if listnet_transformation == 1:
            mat.append((p_y_true_y_true - p_y_true_y_true) ** 2)  # adicionar 0 diretamente?
        elif listnet_transformation == 2:
            mat.append(torch.nn.CosineSimilarity(dim=1)(p_y_true_y_true, p_y_true_y_true))  # adicionar 1 diretamente?
            # mat.append(torch.ones(p_y_true_y_true.shape[0], dtype=torch.float))  # adicionar 1 diretamente?
        elif listnet_transformation == 4:
            mat.append(torch.nn.CosineSimilarity(dim=1)(p_y_true_y_true, p_y_true_y_true))

    mat = torch.stack(mat).to(device)
    if listnet_transformation == 1:
        mat = torch.sum(mat, dim=2)
    mat = mat.t()

    # listnet_transformation==1 deixa como melhor efetividade a maior distância, temos que inverter
    if listnet_transformation == 1:
        mat = -mat + torch.max(mat)
    mat = -mat + torch.max(mat)
    factor = torch.tensor([negative], requires_grad=True, dtype=torch.float, device=device)
    if return_strategy == 1:
        return factor * geoRisk(mat, alpha, device, requires_grad=True)
    elif return_strategy == 2:
        return factor * (geoRisk(mat, alpha, device, requires_grad=True, i=-1) - geoRisk(mat, alpha, device,
                                                                                         requires_grad=True))
    elif return_strategy == 3:
        return factor * ((geoRisk(mat, alpha, device, requires_grad=True, i=-1) - geoRisk(mat, alpha, device,
                                                                                          requires_grad=True)) ** 2)

    return None




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
