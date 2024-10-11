import warnings
from numbers import Integral, Real

from torch import Tensor
import numpy as np
from jaxtyping import Float

from sklearn.decomposition._fastica import _sym_decorrelation

_LN_PI: float = np.log(np.pi)


def _g_log(y, log_h_diag):
    g = -np.log(1 + y) + _LN_PI + 0.5 * log_h_diag
    g_grad = -1 / (1 + y)
    return g, g_grad


def _g_sqrt(y, log_h_diag, alpha=0.8, beta=1.2, epsilon=1e-12):
    g = -alpha * np.sqrt(y + epsilon) + beta + 0.5 * log_h_diag
    g_grad = -alpha / (2 * np.sqrt(y + epsilon))
    return g, g_grad


def _g_square(y, log_h_diag, alpha=0.50, beta=0.50):
    g = alpha * y**2 + beta + 0.5 * log_h_diag
    g_grad = 2 * alpha * y
    return g, g_grad


def _local_energy(X, W, h):
    return h @ ((W @ X) ** 2)


class ExpDecay:
    def __init__(self, lr, decay_rate=0.9999):
        self.lr = lr
        self.t = 0
        self.decay_rate = decay_rate

    def update(self, W, grad):
        self.t += 1
        return self.lr * (self.decay_rate**self.t) * grad


def _tica_par(
    X: Float[Tensor, "n_dims n_samples"],
    h: Float[Tensor, "n_components n_components"],
    max_iter: Integral,
    w_init: Float[Tensor, "n_components n_dims"],
    lr: Real,
    g=_g_sqrt,
    verbose=False,
):
    """Parallel Topographic ICA."""
    W: Float[Tensor, "n_components n_dims"] = w_init
    log_h_diag = np.log(np.diag(h))[:, np.newaxis]
    _, n_samples = X.shape

    W = _sym_decorrelation(W)

    del w_init
    for iter in range(max_iter):
        y = _local_energy(X, W, h)
        # g_y, g_y_grad : Tensor["n_components", "n_samples"]
        g_y, g_y_grad = g(y, log_h_diag)
        # W += lr * np.einsum("jb, il, lb, ik, kb -> ij", X, W, X, h, g_y_grad) / n_samples
        grad = lr * (((W @ X) * (h @ g_y_grad)) @ X.T) / n_samples
        # W += optimizer.update(W, grad)
        W += grad
        W = _sym_decorrelation(W)
        if verbose:
            log_likelihood = g_y.mean()
            print(f"{iter = }, {log_likelihood = }")
            del log_likelihood
        del y, g_y, g_y_grad
    return W


def topographic_ica(
    X: Float[Tensor, "n_dims n_samples"],
    h: Float[Tensor, "n_components n_components"] = None,
    max_iter: Integral = 1000,
    w_init: Float[Tensor, "n_components n_dims"] = None,
    lr: Real = 0.05,
    g: {callable, str} = "sqrt",
    whiten_solver: str = "svd",
    postprocessing: str = "unit-variance",
    verbose: bool = False,
):
    """Topographic ICA.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_dims)
        Training data, where `n_samples` is the number of samples
        and `n_dims` is the number of features.
    h : array-like, shape (n_components, n_components)
        Topographic neighborhood matrix. If None, a neighborhood matrix is
        an identity matrix. (i.e. no topographic constraint)
    max_iter : int
        Maximum number of iterations.
    w_init : array-like, shape (n_components, n_dims)
        Initial un-mixing matrix.
    lr : float
        Learning rate.
    g : {"sqrt", "log", "square", callable}
        The functional form of the G function used in the
        approximation to density function. Could be either 'sqrt', 'square',
        or 'log', or a callable. If callable, it should return a tuple
        containing the value of the function, and of its derivative.
    whiten_solver : {"svd", "eigh", None}
        Solver to use for whitening. If None, no whitening is performed.
    postprocessing : {"unit-variance", "arbitrary-variance"}
        Postprocessing to apply to the estimated sources.
    verbose : bool
        If True, prints log likelihood at each iteration.

    Returns
    -------
    S : array-like, shape (n_samples, n_components)
        Estimated sources.
    components_ : array-like, shape (n_components, n_dims)
        Un-mixing matrix.
    mean_ : array-like, shape (n_dims,)
        Per-feature empirical mean, estimated from the training set.
        Equal to `X.mean(axis=0)`.
    mixing_ : array-like, shape (n_dims, n_components)
        Mixing matrix.
    whitening_ : array-like, shape (n_components, n_dims)
        Whitening matrix.
    unmixing_ : array-like, shape (n_components, n_dims)
        Un-mixing matrix without whitening.
    """

    XT = X.T
    n_dims, n_samples = XT.shape

    if h is not None:
        n_components = h.shape[0]
    else:
        n_components = min(n_samples, n_dims)

    assert h.shape == (n_components, n_components)
    assert w_init.shape == (n_components, n_dims)

    if n_components > min(n_samples, n_dims):
        n_components = min(n_samples, n_dims)
        warnings.warn("n_components is too large: it will be set to %s" % n_components)
    if whiten_solver is not None:
        # Centering the features of X
        X_mean = XT.mean(axis=-1)
        XT -= X_mean[:, np.newaxis]

        # Whitening and preprocessing by PCA
        if whiten_solver == "eigh":
            # Faster when num_samples >> n_features
            d, u = np.linalg.eigh(XT.dot(X))
            sort_indices = np.argsort(d)[::-1]
            eps = np.finfo(d.dtype).eps
            degenerate_idx = d < eps
            if np.any(degenerate_idx):
                warnings.warn(
                    "There are some small singular values, using "
                    "whiten_solver = 'svd' might lead to more "
                    "accurate results."
                )
            d[degenerate_idx] = eps
            np.sqrt(d, out=d)
            d, u = d[sort_indices], u[:, sort_indices]
        elif whiten_solver == "svd":
            u, d = np.linalg.svd(XT, full_matrices=False)[:2]

        # Give consistent eigenvectors for both svd solvers
        u *= np.sign(u[0])

        K = (u / d).T[:n_components]  # see (6.33) p.140
        del u, d
        X1 = np.dot(K, XT)
        # see (13.6) p.267 Here X1 is white and data
        # in X has been projected onto a subspace by PCA
        X1 *= np.sqrt(n_samples)
    else:
        # X must be casted to floats to avoid typing issues with numpy
        # 2.0 and the line below
        X1 = XT.astype(np.float64, copy=False)

    if h is None:
        h = np.eye(n_components)
    else:
        h = np.asarray(h)
        if h.shape != (n_components, n_components):
            raise ValueError(
                "h has invalid shape -- should be %(shape)s"
                % {"shape": (n_components, n_components)}
            )

    if w_init is None:
        w_init = np.random.rand(n_components, n_dims)
    else:
        w_init = np.asarray(w_init)
        if w_init.shape != (n_components, n_dims):
            raise ValueError(
                "w_init has invalid shape -- should be %(shape)s"
                % {"shape": (n_components, n_dims)}
            )

    w_init = w_init @ np.linalg.inv(K)

    if isinstance(g, str):
        if g == "sqrt":
            g = _g_sqrt
        elif g == "log":
            g = _g_log
        elif g == "square":
            g = _g_square
        else:
            raise ValueError("g must be 'sqrt', 'log', or 'square'")

    W = _tica_par(X1, h, max_iter, w_init, lr, g, verbose=verbose)
    del X1

    if whiten_solver is not None:
        S = np.linalg.multi_dot([W, K, XT]).T
    else:
        S = np.dot(W, XT).T

    if postprocessing == "unit-variance":
        S_std = np.std(S, axis=0, keepdims=True)
        S /= S_std
        W /= S_std.T

    components_ = np.dot(W, K)
    mean_ = X_mean
    mixing_ = np.linalg.pinv(components_)
    whitening_ = K
    unmixing_ = W

    return S, components_, mean_, mixing_, whitening_, unmixing_
