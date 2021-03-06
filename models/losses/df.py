import torch
import numpy as np
# from ..differentiable_sorting import diff_sort
# from ..differentiable_sorting import bitonic_matrices as np_bitonic_matrices
### Softmax (log-sum-exp)
def bitonic_layer_loop(n):
    """Outer loop of a bitonic sort, which
    iterates over the sublayers of a bitonic network"""
    layers = int(np.log2(n))
    for layer in range(layers):
        for s in range(layer + 1):
            m = 1 << (layer - s)
            yield n, m, layer


def bitonic_swap_loop(n, m, layer):
    """Inner loop of a bitonic sort,
    which yields the elements to be swapped"""
    out = 0
    for i in range(0, n, m << 1):
        for j in range(m):
            ix = i + j
            a, b = ix, ix + m
            swap = (ix >> (layer + 1)) & 1
            yield a, b, out, swap
            out += 1


def bitonic_network(n):
    """Check the computation of a bitonic network, by printing
    the swapping layers, one permutation per line,
    and a divider after each complete layer block"""
    for n, m, layer in bitonic_layer_loop(n):
        for a, b, out, swap in bitonic_swap_loop(n, m, layer):
            direction = "<" if swap else ">"
            print(f"{a:>2}{direction}{b:<d}", end="\t")
        print()
        if m == 1:
            print("-" * n * 4)
            last_layer = layer


def pretty_bitonic_network(n):
    """Pretty print a bitonic network,
    to check the logic is correct"""
    layers = int(np.log2(n))
    # header
    for i in range(n):
        print(f" {i:<2d}", end="")
    print()

    for n, m, layer in bitonic_layer_loop(n):
        for a, b, out, swap in bitonic_swap_loop(n, m, layer):
            # this could be neater...
            for k in range(n):
                if a == k:
                    if swap:
                        print(" ╰─", end="")
                    else:
                        print(" ╭─", end="")
                elif b == k:
                    if swap:
                        print("─╮ ", end="")
                    else:
                        print("─╯ ", end="")
                elif a < k < b:
                    print("───", end="")
                else:
                    print(" │ ", end="")
            print()
def softmax(a, b, alpha=1, normalize=0):
    """The softmaximum of softmax(a,b) = log(e^a + a^b).
    normalize should be zero if a or b could be negative and can be 1.0 (more accurate)
    if a and b are strictly positive.
    Also called \alpha-quasimax:
            J. Cook.  Basic properties of the soft maximum.
            Working Paper Series 70, UT MD Anderson CancerCenter Department of Biostatistics,
            2011. http://biostats.bepress.com/mdandersonbiostat/paper7
    """
    return np.log(np.exp(a * alpha) + np.exp(b * alpha) - normalize) / alpha


### Smooth max
def smoothmax(a, b, alpha=1):
    return (a * np.exp(a * alpha) + b * np.exp(b * alpha)) / (
            np.exp(a * alpha) + np.exp(b * alpha)
    )


### relaxed softmax
def softmax_smooth(a, b, smooth=0):
    """The smoothed softmaximum of softmax(a,b) = log(e^a + a^b).
    With smooth=0.0, is softmax; with smooth=1.0, averages a and b"""
    t = smooth / 2.0
    return np.log(np.exp((1 - t) * a + b * t) + np.exp((1 - t) * b + t * a)) - np.log(
        1 + smooth
    )


def bitonic_matrices(n):
    """Compute a set of bitonic sort matrices to sort a sequence of
    length n. n *must* be a power of 2.

    See: https://en.wikipedia.org/wiki/Bitonic_sorter

    Set k=log2(n).
    There will be k "layers", i=1, 2, ... k

    Each ith layer will have i sub-steps, so there are (k*(k+1)) / 2 sorting steps total.

    For each step, we compute 4 matrices. l and r are binary matrices of size (k/2, k) and
    map_l and map_r are matrices of size (k, k/2).

    l and r "interleave" the inputs into two k/2 size vectors. map_l and map_r "uninterleave" these two k/2 vectors
    back into two k sized vectors that can be summed to get the correct output.

    The result is such that to apply any layer's sorting, we can perform:

    l, r, map_l, map_r = layer[j]
    a, b =  l @ y, r @ y
    permuted = map_l @ np.minimum(a, b) + map_r @ np.maximum(a,b)

    Applying this operation for each layer in sequence sorts the input vector.

    """
    # number of outer layers

    matrices = []
    for n, m, layer in bitonic_layer_loop(n):
        l, r = np.zeros((n // 2, n)), np.zeros((n // 2, n))
        map_l, map_r = np.zeros((n, n // 2)), np.zeros((n, n // 2))
        for a, b, out, swap in bitonic_swap_loop(n, m, layer):
            l[out, a] = 1
            r[out, b] = 1
            if swap:
                a, b = b, a
            map_l[a, out] = 1
            map_r[b, out] = 1
        matrices.append((l, r, map_l, map_r))
    return matrices


def bitonic_indices(n):
    """Compute a set of bitonic sort indices to sort a sequence of
    length n. n *must* be a power of 2. As opposed to the matrix
    operations, this requires only two index vectors of length n
    for each layer of the network.

    """
    # number of outer layers
    layers = int(np.log2(n))
    indices = []
    for n, m, layer in bitonic_layer_loop(n):
        weave = np.zeros(n, dtype="i4")
        unweave = np.zeros(n, dtype="i4")
        for a, b, out, swap in bitonic_swap_loop(n, m, layer):
            weave[out] = a
            weave[out + n // 2] = b
            if swap:
                a, b = b, a
            unweave[a] = out
            unweave[b] = out + n // 2
        indices.append((weave, unweave))
    return indices


def bitonic_woven_matrices(n):
    """
    Combine the l,r and l_inv, r_inv matrices into single n x n multiplies, for
    use with bisort_weave/diff_bisort_weave, fusing together consecutive stages.
    This reduces the number of multiplies to (k)(k+1) + 1 multiplies, where k=np.log2(n)
    """
    layers = int(np.log2(n))
    matrices = []
    last_unweave = np.eye(n)
    for n, m, layer in bitonic_layer_loop(n):
        weave, unweave = np.zeros((n, n)), np.zeros((n, n))
        for a, b, out, swap in bitonic_swap_loop(n, m, layer):
            weave[out, a] = 1
            weave[out + n // 2, b] = 1
            # flip comparison order as needed
            if swap:
                a, b = b, a
            unweave[a, out] = 1
            unweave[b, out + n // 2] = 1
        # fuse the unweave and weave steps
        matrices.append(weave @ last_unweave)
        last_unweave = unweave
    # make sure the last unweave is preserved
    matrices.append(last_unweave)
    return matrices


def diff_sort(matrices, x, softmax=softmax):
    """
    Approximate differentiable sort. Takes a set of bitonic sort matrices generated by bitonic_matrices(n), sort
    a sequence x of length n. Values may be distorted slightly but will be ordered.
    """
    for l, r, map_l, map_r in matrices:
        a, b = l @ x, r @ x
        mx = softmax(a, b)
        mn = a + b - mx
        x = map_l @ mn + map_r @ mx

    return x


def diff_sort_indexed(indices, x, softmax=softmax):
    """
    Given a set of bitonic sort indices generated by bitonic_indices(n), sort
    a sequence x of length n.
    """
    split = len(x) // 2
    for weave, unweave in indices:
        woven = x[weave]
        a, b = woven[:split], woven[split:]
        mx = softmax(a, b)
        mn = a + b - mx
        x = np.concatenate([mn, mx])[unweave]
    return x


def comparison_sort(matrices, x, compare_fn, alpha=1, scale=250):
    """
    Sort a tensor X, applying a differentiable comparison function "compare_fn"
    while sorting. Uses softmax to weight components of the matrix.

    Parameters:
    ------------
        matrices:   the nxn bitonic sort matrices created by bitonic_matrices
        X:          an [n,...] tensor of elements
        compare_fn: a differentiable comparison function compare_fn(a,b)
                    taking a pair of [n//2,...] tensors and returning a signed [n//2] vector.
        alpha=1.0:  smoothing to apply; smaller alpha=smoother, less accurate sorting,
                    larger=harder max, increased numerical instability
        scale=250:  scaling applied to output of compare_fn. Default is useful for
                    comparison functions returning values in the range ~[-1, 1]

    Returns:
    ----------
        X_sorted: [n,...] tensor (approximately) sorted accoring to compare_fn

    """
    for l, r, map_l, map_r in matrices:
        score = compare_fn((x.T @ l.T).T, (x.T @ r.T).T)
        a, b = score * scale, score * -scale
        a_weight = np.exp(a * alpha) / (np.exp(a * alpha) + np.exp(b * alpha))
        b_weight = 1 - a_weight
        # apply weighting to the full vectors
        aX = x.T @ l.T
        bX = x.T @ r.T
        w_max = (a_weight * aX + b_weight * bX)
        w_min = (b_weight * aX + a_weight * bX)
        # recombine into the full vector
        x = (w_max @ map_l.T) + (w_min @ map_r.T)
        x = x.T

    return x


def vector_sort(matrices, X, key, alpha=1):
    """
    Sort a matrix X, applying a differentiable function "key" to each vector
    while sorting. Uses softmax to weight components of the matrix.

    For example, selecting the nth element of each vector by
    multiplying with a one-hot vector.

    Parameters:
    ------------
        matrices:   the nxn bitonic sort matrices created by bitonic_matrices
        X:          an [n,d] matrix of elements
        key:        a function taking a d-element vector and returning a scalar
        alpha=1.0:  smoothing to apply; smaller alpha=smoother, less accurate sorting,
                    larger=harder max, increased numerical instability

    Returns:
    ----------
        X_sorted: [n,d] matrix (approximately) sorted accoring to

    """
    for l, r, map_l, map_r in matrices:
        x = key(X)
        # compute weighting on the scalar function
        a, b = l @ x, r @ x
        a_weight = np.exp(a * alpha) / (np.exp(a * alpha) + np.exp(b * alpha))
        b_weight = 1 - a_weight
        # apply weighting to the full vectors
        aX = l @ X
        bX = r @ X
        w_max = (a_weight * aX.T + b_weight * bX.T).T
        w_min = (b_weight * aX.T + a_weight * bX.T).T
        # recombine into the full vector
        X = (map_l @ w_max) + (map_r @ w_min)
    return X


def diff_sort_weave(fused, x, softmax=softmax, beta=0.0):
    """
    Given a set of bitonic sort matrices generated by bitonic_woven_matrices(n), sort
    a sequence x of length n.
    beta specifies interpolation between true permutations (beta=0.0) and
    leaving the values unchanged (beta=1.0)
    """
    i = np.eye(len(x))
    split = len(x) // 2
    x = ((beta * i) + (1 - beta) * fused[0]) @ x
    for mat in fused[1:]:
        a, b = x[:split], x[split:]
        mx = softmax(a, b)
        mn = a + b - mx
        x = (beta * i + (1 - beta) * mat) @ np.concatenate([mn, mx])
    return x


### differentiable ranking
def order_matrix(original, sortd, sigma=0.1):
    """Apply a simple RBF kernel to the difference between original and sortd,
    with the kernel width set by sigma. Normalise each row to sum to 1.0."""
    diff = ((original).reshape(-1, 1) - sortd.reshape(1, -1)) ** 2
    rbf = np.exp(-(diff) / (2 * sigma ** 2))
    return (rbf.T / np.sum(rbf, axis=1)).T


def dargsort(original, sortd, sigma, transpose=False):
    """Take an input vector `original` and a sorted vector `sortd`
    along with an RBF kernel width `sigma`, return an approximate ranking.
    If transpose is True, returns approximate argsort (but note that ties have identical values)
    If transpose is False (default), returns ranking"""
    order = order_matrix(original, sortd, sigma=sigma)
    if transpose:
        order = order.T
    return order @ np.arange(len(original))


def diff_argsort(matrices, x, sigma=0.1, softmax=softmax, transpose=False):
    """Return the smoothed, differentiable ranking of each element of x. Sigma
    specifies the smoothing of the ranking. Note that this function is deceptively named,
    and in the default setting returns the *ranking*, not the argsort.

    If transpose is True, returns argsort (but note that ties are not broken in differentiable
    argsort);
    If False, returns ranking (likewise, ties are not broken).
    """
    sortd = diff_sort(matrices, x, softmax)
    return dargsort(x, sortd, sigma, transpose)


def diff_argsort_indexed(indices, x, sigma=0.1, softmax=softmax, transpose=False):
    """Return the smoothed, differentiable ranking of each element of x. Sigma
    specifies the smoothing of the ranking. Uses the indexed form
    to avoid multiplies.

    If transpose is True, returns argsort (but note that ties are not broken in differentiable
    argsort);
    If False, returns ranking (likewise, ties are not broken).
    """
    sortd = diff_sort_indexed(indices, x, softmax)
    return dargsort(x, sortd, sigma, transpose)

def np_bitonic_matrices(n):
    """Compute a set of bitonic sort matrices to sort a sequence of
    length n. n *must* be a power of 2.

    See: https://en.wikipedia.org/wiki/Bitonic_sorter

    Set k=log2(n).
    There will be k "layers", i=1, 2, ... k

    Each ith layer will have i sub-steps, so there are (k*(k+1)) / 2 sorting steps total.

    For each step, we compute 4 matrices. l and r are binary matrices of size (k/2, k) and
    map_l and map_r are matrices of size (k, k/2).

    l and r "interleave" the inputs into two k/2 size vectors. map_l and map_r "uninterleave" these two k/2 vectors
    back into two k sized vectors that can be summed to get the correct output.

    The result is such that to apply any layer's sorting, we can perform:

    l, r, map_l, map_r = layer[j]
    a, b =  l @ y, r @ y
    permuted = map_l @ np.minimum(a, b) + map_r @ np.maximum(a,b)

    Applying this operation for each layer in sequence sorts the input vector.

    """
    # number of outer layers

    matrices = []
    for n, m, layer in bitonic_layer_loop(n):
        l, r = np.zeros((n // 2, n)), np.zeros((n // 2, n))
        map_l, map_r = np.zeros((n, n // 2)), np.zeros((n, n // 2))
        for a, b, out, swap in bitonic_swap_loop(n, m, layer):
            l[out, a] = 1
            r[out, b] = 1
            if swap:
                a, b = b, a
            map_l[a, out] = 1
            map_r[b, out] = 1
        matrices.append((l, r, map_l, map_r))
    return matrices



### Softmax (log-sum-exp)
def softmax(a, b, alpha=1.0, normalize=0.0):
    """The softmaximum of softmax(a,b) = log(e^a + a^b).
    normalize should be zero if a or b could be negative and can be 1.0 (more accurate)
    if a and b are strictly positive.
    """
    return torch.log(torch.exp(a * alpha) + torch.exp(b * alpha) - normalize) / alpha


### Smooth max
def smoothmax(a, b, alpha=1.0):
    return (a * torch.exp(a * alpha) + b * torch.exp(b * alpha)) / (
            torch.exp(a * alpha) + torch.exp(b * alpha)
    )


### relaxed softmax
def softmax_smooth(a, b, smooth=0.0):
    """The smoothed softmaximum of softmax(a,b) = log(e^a + a^b).
    With smooth=0.0, is softmax; with smooth=1.0, averages a and b"""
    t = smooth / 2.0
    return torch.log(
        torch.exp((1.0 - t) * a + b * t) + torch.exp((1.0 - t) * b + t * a)
    ) - np.log(1.0 + smooth)


### differentiable ranking
def order_matrix(original, sortd, sigma=0.1):
    """Apply a simple RBF kernel to the difference between original and sortd,
    with the kernel width set by sigma. Normalise each row to sum to 1.0."""
    diff = (original.reshape(-1, 1) - sortd) ** 2
    rbf = torch.exp(-(diff) / (2 * sigma ** 2))
    return (rbf.t() / torch.sum(rbf, dim=1)).t()


def diff_argsort(matrices, x, sigma=0.1, softmax=softmax, transpose=False):
    """Return the smoothed, differentiable ranking of each element of x. Sigma
    specifies the smoothing of the ranking. """
    sortd = diff_sort(matrices, x, softmax)
    order = order_matrix(x, sortd, sigma=sigma)
    if transpose:
        order = order.t()
    return order @ (torch.arange(len(x), dtype=x.dtype))


def bitonic_matrices(n):
    matrices = np_bitonic_matrices(n)
    return [
        [torch.from_numpy(matrix).float() for matrix in matrix_set]
        for matrix_set in matrices
    ]


def vector_sort(matrices, X, key, alpha=1):
    """
    Sort a matrix X, applying a differentiable function "key" to each vector
    while sorting. Uses softmax to weight components of the matrix.

    For example, selecting the nth element of each vector by
    multiplying with a one-hot vector.

    Parameters:
    ------------
        matrices:   the nxn bitonic sort matrices created by bitonic_matrices
        X:          an [n,d] matrix of elements
        key:        a function taking a d-element vector and returning a scalar
        alpha=1.0:  smoothing to apply; smaller alpha=smoother, less accurate sorting,
                    larger=harder max, increased numerical instability

    Returns:
    ----------
        X_sorted: [n,d] matrix (approximately) sorted accoring to

    """
    for l, r, map_l, map_r in matrices:
        x = key(X)
        # compute weighting on the scalar function
        a, b = l @ x, r @ x
        a_weight = torch.exp(a * alpha) / (torch.exp(a * alpha) + torch.exp(b * alpha))
        b_weight = 1 - a_weight
        # apply weighting to the full vectors
        aX = l @ X
        bX = r @ X
        w_max = (a_weight * aX.t() + b_weight * bX.t()).t()
        w_min = (b_weight * aX.t() + a_weight * bX.t()).t()
        # recombine into the full vector
        X = (map_l @ w_max) + (map_r @ w_min)
    return