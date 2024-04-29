# TODO: internal text embedding class/model, course ranking

from sklearn.base import BaseEstimator, TransformerMixin

class RBF(BaseEstimator, TransformerMixin):
    """ Sets up a set of radial basis functions over a given
    range. The equation of the kth radial basis function is:

    f_k(x) = exp(-psi * (x - c[k]) ** 2)

    where c[k] is the 'center' of the kth basis function, meaning
    the value where f_k(x) = 1.

    Parameters
    ----------
    lo: float
        The minimum value that is expected as input.
    hi: float
        The maximum value that is expected as input.
    num_basis_functions: int
        The number of radial basis functions. These will be
        linearly spaced over the given range.
    sigma: float (default = 1.)
        The number of inter-center gaps before the basis function value
        shrinks to 1/e.

    Attributes
    ----------
    c_: ndarray
        The locations of the RBF centers.
    psi_: float
        The constant in the exponential. This is derived from lo, hi, and sigma.

    """
    def __init__(self, lo, hi, num_basis_functions, sigma = 1.):
        self.lo = lo
        self.hi = hi
        self.num_basis_functions = num_basis_functions
        self.sigma = sigma

    def fit(self, X = None, Y = None):
        """ Fits this set of RBF functions. No data necessary. """
        if self.num_basis_functions <= 1:
            self.c_ = np.array([0.5 * (self.lo + self.hi)])
            delta   = 0.5 * (self.hi - self.lo)
        else:
            self.c_   = np.linspace(self.lo, self.hi, self.num_basis_functions)
            delta     = (self.hi - self.lo) / (self.num_basis_functions - 1)
        self.psi_ = 1. / (delta * self.sigma) ** 2
        return self

    def transform(self, X):
        """ Applies the radial basis functions to all dimensions of the given input matrix.

        Parameters
        ----------
        X: ndarray
            An m x n matrix of input values.

        Returns
        -------
        Phi: ndarray
            An m x n x self.num_basis_functions tensor of RBF values, representing the given input.

        """
        cexpand = self.c_
        for dim in range(len(X.shape)):
            cexpand = np.expand_dims(cexpand, 0)
        Phi = np.expand_dims(X, -1) - cexpand
        Phi = np.exp(-self.psi_ * np.square(Phi))
        return Phi


class Sigmoids(BaseEstimator, TransformerMixin):
    """ Sets up a set of sigmoid basis functions over a given
    range. The equation of the kth basis function is:

    f_k(x) = 1 / (1 + exp(-beta * (x - c[k])))

    where c[k] is the 'center' of the kth basis function, meaning
    the value where f_k(x) = .5.

    Parameters
    ----------
    lo: float
        The minimum value that is expected as input.
    hi: float
        The maximum value that is expected as input.
    num_basis_functions: int
        The number of radial basis functions. These will be
        linearly spaced over the given range.
    sigma: float (default = 1.)
        The number of inter-center gaps before the basis function value
        shrinks to 1/(e + 1).

    Attributes
    ----------
    c_: ndarray
        The locations of the RBF centers.
    beta_: float
        The constant in the exponential. This is derived from lo, hi, and sigma.

    """
    def __init__(self, lo, hi, num_basis_functions, sigma = 1.):
        self.lo = lo
        self.hi = hi
        self.num_basis_functions = num_basis_functions
        self.sigma = sigma

    def fit(self, X = None, Y = None):
        """ Fits this set of sigmoid functions. No data necessary. """
        if self.num_basis_functions <= 1:
            self.c_ = np.array([0.5 * (self.lo + self.hi)])
            delta   = 0.5 * (self.hi - self.lo)
        else:
            self.c_   = np.linspace(self.lo, self.hi, self.num_basis_functions)
            delta     = (self.hi - self.lo) / (self.num_basis_functions - 1)
        self.beta_ = 1. / (delta * self.sigma)
        return self

    def transform(self, X):
        """ Applies the radial basis functions to all dimensions of the given input matrix.

        Parameters
        ----------
        X: ndarray
            An m x n matrix of input values.

        Returns
        -------
        Phi: ndarray
            An m x n x self.num_basis_functions tensor of RBF values, representing the given input.

        """
        cexpand = self.c_
        for dim in range(len(X.shape)):
            cexpand = np.expand_dims(cexpand, 0)
        Phi = np.expand_dims(X, -1) - cexpand
        Phi = 1. / (1. + np.exp(-self.beta_ * Phi))
        return Phi