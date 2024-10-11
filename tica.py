
from sklearn.base import BaseEstimator, TransformerMixin, _ClassNamePrefixFeaturesOutMixin
from numpy_tica import topographic_ica as topographic_ica
_backend = []

_backend.append('numpy')


class TopographicICA(_ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):

    def __init__(
        self,
        n_components=None,
        *,
        fun="sqrt",
        fun_args=None,
        max_iter=1000,
        w_init=None,
        lr=0.05,
        whiten_solver="svd",
        postprocessing="unit-variance",
        verbose=False,
    ):
        super().__init__()
        self.n_components = n_components
        self.fun = fun
        self.fun_args = fun_args
        self.max_iter = max_iter
        self.w_init = w_init
        self.lr = lr
        self.whiten_solver = whiten_solver
        self.postprocessing = postprocessing
        self.verbose = verbose

    def _check_array_type(self, X):
        type_str = str(type(X))
        if 'numpy' in type_str:
            return 'numpy'
        else:
            return type_str

    def _fit_transform(self, X, h=None):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        h : array-like of shape (n_components, n_components)
            Topographic neighborhood matrix. If None, a neighborhood matrix is
            an identity matrix. (i.e. no topographic constraint)

        Returns
        -------
        S : ndarray of shape (n_samples, n_components) or None
            Sources matrix. `None` if `compute_sources` is `False`.
        """

        array_type = self._check_array_type(X)
        if array_type == 'numpy':
            f = topographic_ica
        else:
            raise ValueError(f"Unknown array type: {array_type}")

        S, self.components_, self.mean_, \
            self.mixing_, self.whitening_, self.unmixing_ = f(
                X,
                h=h,
                g=self.fun,
                # fun_args=self.fun_args,
                max_iter=self.max_iter,
                w_init=self.w_init,
                lr=self.lr,
                whiten_solver=self.whiten_solver,
                postprocessing=self.postprocessing,
                verbose=self.verbose,
            )

        return S

    def fit_transform(self, X, h=None):
        """Fit the model and recover the sources from X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        h : array-like of shape (n_components, n_components)
            Topographic neighborhood matrix. If None, a neighborhood matrix is
            an identity matrix. (i.e. no topographic constraint)

        Returnsf
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Estimated sources obtained by transforming the data with the
            estimated unmixing matrix.
        """

        return self._fit_transform(X, h)

    def fit(self, X, h=None):
        """Fit the model to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        h : array-like of shape (n_components, n_components)
            Topographic neighborhood matrix. If None, a neighborhood matrix is
            an identity matrix. (i.e. no topographic constraint)

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        self._fit_transform(X, h)
        return self

    def available_backends(self):
        return _backend
