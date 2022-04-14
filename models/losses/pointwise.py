import torch

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import DEFAULT_EPS
from allrank.models.metrics import ndcg


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


def pointwise_rmse(y_pred, y1, y_true, no_of_levels=5, normalized=1, padded_value_indicator=PADDED_Y_VALUE, ob=2):
    """
    Pointwise RMSE loss.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param no_of_levels: number of unique ground truth values
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    if ob != 2:
        ############
        mat = []
        mat.append(torch.squeeze(ndcg(y_pred, y_true)))
        for i in y1:
            mat.append(torch.squeeze(ndcg(i, y_true)))
        mat = torch.stack(mat).to(y_pred.device)
        device = y_pred.device
        mat = mat.t()
        selected_grisk = geoRisk(mat, 5.0, device, requires_grad=True)
        index = 0
        for i in range(mat.shape[1]):
            u = geoRisk(mat, 5.0, device, requires_grad=True, i=i)
            if ob == 1:
                if u > selected_grisk:
                    selected_grisk = u
                    index = i
            else:
                if u < selected_grisk:
                    selected_grisk = u
                    index = i
        if index != 0:
            y_pred = y1[index - 1]

        ##############

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

    return torch.mean(rmses)


def pointwise_rmsemulti(y_pred, y1, y_true, no_of_levels=5, normalized=1, padded_value_indicator=PADDED_Y_VALUE, ob=2):
    soma = 0
    if isinstance(y1, list):

        for i in y1:
            soma += pointwise_rmse(i, i, y_true, no_of_levels, normalized, padded_value_indicator, 2)
            # i, y_true
    else:
        for i in range(y1.shape[2]):
            soma += pointwise_rmse(y1[:, :, i], y1[:, :, i], y_true, no_of_levels, normalized, padded_value_indicator,
                                   2)

            # y1[:, :, i], y_true
    return soma
