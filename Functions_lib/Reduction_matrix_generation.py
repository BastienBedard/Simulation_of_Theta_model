import numpy as np
from numpy.linalg import pinv, det
# Note: I was not able to jit from numba for snmf and onmf to get reduction mat

import logging
import logging.config
import scipy.sparse


_EPS = np.finfo(float).eps


class PyMFBase:
    """
    PyMF Base Class. Does nothing useful apart from providing
    some basic methods.
    """

    _EPS = _EPS  # some small value

    def __init__(self, data, W_init=None, H_init=None, num_bases=4):
        """
        """

        def setup_logging():
            # create logger
            self._logger = logging.getLogger("pymf")

            # add ch to logger
            if len(self._logger.handlers) < 1:
                # create console handler and set level to debug
                ch = logging.StreamHandler()
                ch.setLevel(logging.DEBUG)
                # create formatter
                formatter = logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(message)s")

                # add formatter to ch
                ch.setFormatter(formatter)

                self._logger.addHandler(ch)

        setup_logging()

        # set variables
        self.data = data
        self._num_bases = num_bases

        self.W_init = W_init
        self.H_init = H_init

        self._data_dimension, self._num_samples = self.data.shape

    def residual(self):
        """ Returns the residual in % of the total amount of data
        Returns
        -------
        residual : float
        """
        res = np.sum(np.abs(self.data - np.dot(self.W, self.H)))
        total = 100.0 * res / np.sum(np.abs(self.data))
        return total

    def frobenius_norm(self):
        """ Frobenius norm (||data - WH||) of a data matrix and a low rank
        approximation given by WH. Minimizing the Fnorm is the most common
        optimization criterion for matrix factorization methods.
        Returns:
        -------
        frobenius norm: F = ||data - WH||
        """
        # check if W and H exist
        if hasattr(self, 'H') and hasattr(self, 'W'):
            if scipy.sparse.issparse(self.data):
                tmp = self.data[:, :] - (self.W * self.H)
                tmp = tmp.multiply(tmp).sum()
                err = np.sqrt(tmp)
            else:
                err = np.sqrt(
                    np.sum((self.data[:, :] - np.dot(self.W, self.H)) ** 2))
        else:
            err = None

        return err

    def _init_w(self):
        """ Initalize W to random values in [0,1] if W_init is None.
            Else it initializes to the given W_init matrix.
        """
        if self.W_init is None:
            # add a small value, otherwise nmf and related methods get
            # into trouble as
            # they have difficulties recovering from zero.
            self.W = np.random.random(
                (self._data_dimension, self._num_bases)) + 10 ** -4
        else:
            self.W = self.W_init

    def _init_h(self):
        """ Initalize H to random values in [0,1] if H_init is None.
            Else it initializes to the given H_init matrix.
        """
        if self.H_init is None:
            # add a small value, otherwise nmf and related methods get
            # into trouble as
            # they have difficulties recovering from zero.
            self.H = np.random.random(
                (self._num_bases, self._num_samples)) + 10**(-4)
        else:
            self.H = self.H_init

    def _update_h(self):
        """ Overwrite for updating H.
        """
        pass

    def _update_w(self):
        """ Overwrite for updating W.
        """
        pass

    def _converged(self, i):
        """
        If the optimization of the approximation is below the machine precision
        return True.
        Parameters
        ----------
            i   : index of the update step
        Returns
        -------
            converged : boolean
        """
        derr = np.abs(self.ferr[i] - self.ferr[i - 1]) / self._num_samples
        if derr < self._EPS:
            return True
        else:
            return False

    def factorize(self, niter=100,  # show_progress=False,
                  compute_w=True, compute_h=True, compute_err=True):
        """ Factorize s.t. WH = data

        Parameters
        ----------
        niter : int
                number of iterations.
        # show_progress : bool
        #         print some extra information to stdout.
        compute_h : bool
                iteratively update values for H.
        compute_w : bool
                iteratively update values for W.
        compute_err : bool
                compute Frobenius norm |data-WH| after each update and store
                it to .ferr[k].

        Updated Values
        --------------
        .W : updated values for W.
        .H : updated values for H.
        .ferr : Frobenius norm |data-WH| for each iteration.
        """

        # if show_progress:
        #     self._logger.setLevel(logging.INFO)
        # else:
        #     self._logger.setLevel(logging.ERROR)

        # create W and H if they don't already exist
        # -> any custom initialization to W, H should be done before
        if not hasattr(self, 'W') and compute_w:
            self._init_w()

        if not hasattr(self, 'H') and compute_h:
            self._init_h()

        # Computation of the error can take quite long for large matrices,
        # thus we make it optional.
        if compute_err:
            self.ferr = np.zeros(niter)

        for i in range(niter):
            if compute_w:
                self._update_w()

            if compute_h:
                self._update_h()

            if compute_err:
                self.ferr[i] = self.frobenius_norm()
                # self._logger.info(
                #     'FN: %s (%s/%s)' % (self.ferr[i], i + 1, niter))
            # else:
            #     # self._logger.info('Iteration: (%s/%s)' % (i + 1, niter))

            # check if the err is not changing anymore
            if i > 1 and compute_err:
                if self._converged(i):
                    # adjust the error measure
                    self.ferr = self.ferr[:i]
                    break


__all__ = ["SNMF"]


class SNMF(PyMFBase):
    """
    SNMF(data, H_init=H_init, W_init=W_init, num_bases=4)

    Semi Non-negative Matrix Factorization. Factorize a data matrix into two
    matrices s.t. F = | data - W*H | is minimal. For Semi-NMF only H is
    constrained to non-negativity.

    Parameters
    ----------
    - data : array_like, shape (_data_dimension, _num_samples)
        the input data
    - num_bases: int, optional
        Number of bases to compute (column rank of W and row rank of H).
        4 (default)

    Attributes
    ----------
    - W : "data_dimension x num_bases" matrix of basis vectors
    - H : "num bases x num_samples" matrix of coefficients
    - ferr : frobenius norm (after calling .factorize())

    Example
    -------
    Applying Semi-NMF to some rather stupid data set:

    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> snmf_mdl = SNMF(data, num_bases=2)
    >>> snmf_mdl.factorize(niter=10)

    The basis vectors are now stored in snmf_mdl.W, the coefficients in
    snmf_mdl.H.
    To compute coefficients for an existing set of basis vectors simply copy W
    to snmf_mdl.W, and set compute_w to False:

    >>> data = np.array([[1.5], [1.2]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> snmf_mdl = SNMF(data, num_bases=2)
    >>> snmf_mdl.W = W
    >>> snmf_mdl.factorize(niter=1, compute_w=False)

    The result is a set of coefficients snmf_mdl.H, s.t. data = W * snmf_mdl.H.
    """

    def _update_w(self):
        W1 = np.dot(self.data[:, :], self.H.T)
        W2 = np.dot(self.H, self.H.T)
        # if np.abs(np.linalg.det(W2)) < 1e-8:
        #     raise ValueError("A matrix in the snmf (W2) is singular !")
        # else:
        #     self.W = np.dot(W1, np.linalg.inv(W2))

    def _update_h(self):
        def separate_positive(m):
            return (np.abs(m) + m) / 2.0

        def separate_negative(m):
            return (np.abs(m) - m) / 2.0

        XW = np.dot(self.data[:, :].T, self.W)

        WW = np.dot(self.W.T, self.W)
        WW_pos = separate_positive(WW)
        WW_neg = separate_negative(WW)

        XW_pos = separate_positive(XW)
        H1 = (XW_pos + np.dot(self.H.T, WW_neg)).T

        XW_neg = separate_negative(XW)
        H2 = (XW_neg + np.dot(self.H.T, WW_pos)).T + 10 ** -9

        self.H *= np.sqrt(H1 / H2)

# -------------------------- Matrix conditions --------------------------------


def matrix_is_singular(C):
    boolean = 0
    if np.abs(det(C)) < 1e-8:
        boolean = 1
    return boolean


def matrix_is_negative(M):
    return np.any(M < -1e-8)


def matrix_has_rank_n(M):
    n = len(M[:, 0])
    return np.linalg.matrix_rank(M) == n


def matrix_is_normalized(M):
    if len(np.shape(M)) == 1:
        bool_value = np.absolute(np.sum(M) - 1) < 0.000001
    else:
        n = len(M[:, 0])
        bool_value = \
            np.all(np.absolute(np.sum(M, axis=1) - np.ones(n)) < 0.000001)
    return bool_value


def matrix_is_orthogonal(M):
    boolean = 0
    X = np.abs(M@M.T - np.identity(np.shape(M)[0]))
    Y = np.abs(X - np.diag(np.diag(X)))
    # Because we don't want to check normalization, we substract the diagonal
    # of X to X to get Y. If it is a zero matrix, then M is orthogonal.
    if np.all(Y < 1e-8):
        boolean = 1
    return boolean


def matrix_is_orthonormalized_VV_T(V):
    n = len(V[:, 0])
    return np.all(np.absolute(V@V.T - np.identity(n)) < 1e-8)


def matrix_is_positive(M):
    return np.all(M >= -1e-8)


# ----------------- Algorithms to get the reduced matrices --------------------

def get_second_target_coefficent_matrix(V_T1, V_T2, V_T3,
                                        other_procedure=True):
    """
    This function is useful in a three-target procedure.
    :param V_T1: First target eigenvector matrix
    :param V_T2: Second target eigenvector matrix or a null matrix
                 np.zeros((n, N)) if it is not a two target procedure
    :param V_T3: Third target eigenvector matrix or a null matrix
                 np.zeros((n, N)) if it is not a three target procedure

    :param other_procedure: If True, it is the procedure 4 in the sync article
                            If False, it is the procedure 3 in the sync article
                            The name of the procedure will probably change
                            in the article,

    :return:
    """
    n, N = np.shape(V_T1)

    if np.all(V_T3 < 1e-10):
        C_T2 = np.zeros((n, n))
        # See the documentation of the function
        # get_first_target_coefficent_matrix(C_T2, V_T1, V_T2)

    else:
        if other_procedure:
            C_T2 = V_T3@pinv(V_T1)@V_T1@pinv(V_T2)
        else:
            C_T2 = V_T3@pinv(V_T2)

    return C_T2


def get_first_target_coefficent_matrix(C_T2, V_T1, V_T2):
    """
    :param V_T1: First target eigenvector matrix
    :param V_T2: Second target eigenvector matrix
    :param C_T2: Second_target_coefficent_matrix
                 without the normalization (C_T3 would be the normalization
                 matrix)

                 Can either be

                 1 - np.zeros((n, n)) for a two-target procedure
                 which only means that C_T2 will be use
                 to respect condition B (normalization and non negativity)

                 2 - a square matrix obtained with
                 get_second_target_coefficent_matrix
                 for a three-target procedure

    :return: C_T1:Coefficent matrix for target 1
                  without the normalization (it is related to C_T2 in a three
                  target procedure. In a two target procedure, C_T2 is related
                  to the normalization)
    """
    n = len(V_T1[:, 0])
    if np.all(C_T2 == np.zeros((n, n))):
        C_T1 = V_T2@pinv(V_T1)
    else:
        C_T1 = C_T2@V_T2@pinv(V_T1)
    return C_T1


def snmf(M, niter=500, W_init=None, H_init=None):
    """
    SNMF: Semi-nonnegative matrix factorization
    :param M: n x N matrix (n > N)
    :param niter: number of iteration in the algorithm, 100 iterations is a
                  safe number of iterations, see Ding 2010
    :param H_init:
    :param W_init:
    :return:
    """
    n, N = np.shape(M)
    snmf_mdl = SNMF(M, H_init=H_init, W_init=W_init, num_bases=n)
    snmf_mdl.factorize(niter=niter)
    # ---------------------------- Normalized frobenius error
    return snmf_mdl.W, snmf_mdl.H, snmf_mdl.ferr[-1]/(n*N)**2


def snmf_multiple_inits(M, number_initializations):
    """
    Notation: W -> F   and   H -> G
    :param M:
    :param number_initializations:
    :return:
    """
    n, N = np.shape(M)

    """ ---------------------------- SVD ---------------------------------- """
    u, s, vh = np.linalg.svd(M)

    # Initial matrix H with SVD
    G_init = np.absolute(vh[0:n, :])

    # Semi nonnegative matrix factorization
    # with SVD initialization
    F_svd, G_svd, frobenius_error_svd = snmf(M, H_init=G_init)
    # if not matrix_is_singular(F_svd):
    F, G = F_svd, G_svd
    snmf_frobenius_error = frobenius_error_svd
    # print("snmf_frobenius_error_svd = ", snmf_frobenius_error)
    # if matrix_is_singular(F):
    #     for j in range(number_initializations):
    #         # Semi nonnegative matrix factorization
    #         # with random initialization
    #         F_random, G_random, frobenius_error_random = snmf(M, H_init=None)
    #         print(det(F_random))
    #         if not matrix_is_singular(F_random):
    #             F, G, = F_random, G_random
    #             snmf_frobenius_error = frobenius_error_random
    #
    # else:

    """ -------------------------- Random --------------------------------- """
    for j in range(number_initializations):
        # Semi nonnegative matrix factorization
        # with random initialization
        F_random, G_random, frobenius_error_random = snmf(M, H_init=None)
        # print(det(F_random))
        if snmf_frobenius_error > frobenius_error_random:
            F, G, = F_random, G_random
            snmf_frobenius_error = frobenius_error_random
            # print("snmf_frobenius_error_random = ", snmf_frobenius_error)

    # print("snmf_frobenius_error_svd = ", snmf_frobenius_error)

    if matrix_is_singular(F):
        raise ValueError("W is singular in the semi-nonnegative matrix"
                         " factorization (snmf).")
    # ---------- Normalized frobenius error
    return F, G, snmf_frobenius_error


def onmf(M, max_iter=500, W_init=None, H_init=None):
    """
    Orthogonal Non-negative Matrix Factorization of X as X =WH wit HH^T=I.
    Based on Ref. Wang, Y. X., & Zhang, Y. J. (2012).
    Nonnegative matrix factorization: A comprehensive review.
    IEEE Transactions on Knowledge and Data Engineering, 25(6), 1336-1353.

    and

    https://github.com/mstrazar/iONMF

    ----------
    Input
    ----------
    M: array [n x N]
        Data matrix to be factorized.
    max_iter: int
        Maximum number of iterations.
    H_init: array [n x n]
        Fixed initial basis matrix.
    W_init: array [n x N]
        Fixed initial coefficient matrix.
    MoreOrtho: Boolean
        If True, searches for a matrix H with more zeros
    ---------
    Output
    ---------
    W: array [n x n]
    H: array [n x N]
    error: ||X-WH||/(nN)^2
        normalized factorization error
    o_error:  ||I-HH^T||/(n^2)
        normalized orthogonality error

    ex: SVD initialization
    n,N = X.shape
    # SVD
    u,s,vh = np.linalg.svd(X)
    # Initial matrix H
    h_init= abs(vh[0:n,:])
    # NMF
    W,H,e,oe = onmf(X, H_init = h_init)

    """

    n, N = np.shape(M)

    # add a small value, otherwise nmf and related methods get
    # into trouble as they have difficulties recovering from zero.
    W = np.random.random((n, n)) + 10**(-4) if isinstance(W_init, type(None))\
        else W_init
    H = np.random.random((n, N)) + 10**(-4) if isinstance(H_init, type(None))\
        else H_init

    for itr in range(max_iter):
        # update H
        numerator = W.T@M
        denominator = H@M.T@W@H
        H = np.nan_to_num(H*numerator/denominator)

        # new lines added to get orthonormalized rows
        row_norm = np.sqrt(np.diag(H@H.T))
        # if matrix_is_singular(np.diag(row_norm)):
        #     plt.imshow(np.diag(row_norm))
        #     plt.colorbar()
        #     plt.show()
        #     print(row_norm)
        #     plt.imshow(M)
        #     plt.colorbar()
        #     plt.show()
        normalization_matrix = np.linalg.inv(np.diag(row_norm))
        H = normalization_matrix@H

        # update W
        numerator = M@H.T
        denominator = W@H@H.T
        W = np.nan_to_num(W*numerator/denominator)

    # error with normalized Frobenius norm
    error = np.linalg.norm(M - W@H)/(n*N)**2

    # orthogonality error with Frobenius norm
    o_error = np.linalg.norm(np.eye(n, n) - H@H.T)/(n**2)

    # ---------- Normalized frobenius error and normalized orthogonal error
    return W, H, error, o_error


def onmf_multiple_inits(M, number_initializations):
    """
    Notation: W -> F   and   H -> G
    :param M:
    :param number_initializations:
    :return:
    """
    n, N = np.shape(M)

    """ ---------------------------- SVD ---------------------------------- """
    u, s, vh = np.linalg.svd(M)

    # Initial matrix H with SVD
    G_init = np.absolute(vh[0:n, :])
    # Ortogonal nonnegative matrix factorization
    # with SVD initialization
    F_svd, G_svd, frobenius_error_svd, ortho_error_svd \
        = onmf(M, H_init=G_init)
    F, G = F_svd, G_svd
    onmf_frobenius_error, onmf_ortho_error = \
        frobenius_error_svd, ortho_error_svd

    # print(f"\nonmf_frobenius_error_svd = {onmf_frobenius_error}",
    #       f"\nonmf_ortho_error_svd = {onmf_ortho_error}")

    """ --------------------------- Random -------------------------------- """
    onmfiter = 0
    while 0.3*onmf_frobenius_error**2 + 0.7*onmf_ortho_error**2 > 1e-7 and onmfiter<number_initializations:
    # for j in range(number_initializations):
        onmfiter+=1
        # Ortogonal nonnegative matrix factorization
        # with random initialization
        F_random, G_random, frobenius_error_random, ortho_error_random \
            = onmf(M, H_init=None)   # S'assurer que c'est ok

        # 1. The condition below is the one used for transitions vs. n in the
        #    reply to the referee. The errors are normalized.
        # if frobenius_error_random**2 + ortho_error_random**2 < \
        #         onmf_frobenius_error**2 + onmf_ortho_error**2:

        # 2. We can penalize the orthogonal errors by adding weights if we want
        # if 0.1*frobenius_error_random**2 + 0.9*ortho_error_random**2 < \
        #         0.1*onmf_frobenius_error**2 + 0.9*onmf_ortho_error**2:
        #

        # 3. The condition below is the one used for FIG. 6 and 7 of the paper
        # if frobenius_error_random < onmf_frobenius_error:

        if 0.3*frobenius_error_random**2 + 0.7*ortho_error_random**2 < \
                0.3*onmf_frobenius_error**2 + 0.7*onmf_ortho_error**2:
            F, G, = F_random, G_random
            onmf_frobenius_error, onmf_ortho_error = \
                frobenius_error_random, ortho_error_random
            # print("Result improved by a random initialization !")
            # print(f"onmf_frobenius_error_random = {onmf_frobenius_error}",
            #       f"\nonmf_ortho_error_random = {onmf_ortho_error}")
        if onmfiter == number_initializations:
            print('max iter onmf')
        if 0.3*onmf_frobenius_error**2 + 0.7*onmf_ortho_error**2 < 1e-7:
            print('complete after', onmfiter)

    if onmfiter == 0:
        print('allready fine')
    if matrix_is_singular(F):
        ValueError('F is singular.')

    # print("onmf_frobenius_error_final = ", onmf_frobenius_error,
    #       "\nonmf_ortho_error_final = ", onmf_ortho_error)

    # ---------- Normalized frobenius error  and normalized ortho errors
    return F, G, onmf_frobenius_error, onmf_ortho_error


def normalize_rows_matrix_M1(M):
    return (M.T / np.sum(M, axis=1)).T


def normalize_rows_complex_matrix_M1(M):
    return (M.T / np.sum(M, axis=1)).T


def normalize_rows_matrix_VV_T(V):
    return (V.T / np.sqrt(np.sum(V**2, axis=1))).T


def get_reduction_matrix(V_T1, V_T2, V_T3, number_initializations=2000,
                         other_procedure=True):
    """
    Get the reduction matrix M for the dimension-reduction.

    :param V_T1: First target eigenvector matrix
    :param V_T2: Second target eigenvector matrix or a null matrix
                 np.zeros((n, N)) if it is not a two target procedure
    :param V_T3: Third target eigenvector matrix or a null matrix
                 np.zeros((n, N)) if it is not a three target procedure
    :param number_initializations:
            Number of different initializations of the semi and the
            orthogonal nonnegative matrix factorization (SNMF and ONMF).
            If niter=1, the algorithm will initialize SNMF and ONMF with SVD.
            If niter>1, the algorithm will initialize SNMF and ONMF with SVD in
            the first iteration and then, it will try random
            initializations to find the lowest Frobenius norm error
            frobenius_error = ||M - WH||.
    :param other_procedure: If True, it is the procedure 4 in the sync article
                            If False, it is the procedure 3 in the sync article
                            The name of the procedure will probably change
                            in the article,

    :return: M : n x N positive array/matrix. np.sum(M[mu,:], axis=1) = 1
                 for all mu which means that the matrix is normalized according
                 to its rows (the sum over the columns is one for each row)
    """

    n, N = np.shape(V_T1)
    # print(f"\nV_T1 = {V_T1}, \n V_T2 = {V_T2}, \n V_T3 = {V_T3}")

    if not np.all(V_T2 == np.zeros((n, N))):
        #Then it is a two or three targets procedure

        op = other_procedure
        C_T2 = get_second_target_coefficent_matrix(V_T1, V_T2, V_T3,
                                                   other_procedure=op)
        C_T1 = get_first_target_coefficent_matrix(C_T2, V_T1, V_T2)

        V = C_T1 @ V_T1

        # print(np.linalg.norm(np.eye(n, n) - V@V.T)/(n**2))
        # print("V_T1 = ", V_T1)
        # print("V_T2 = ", V_T2)
        # print("V = ", V)
        # print(f"\n C_T1 = {C_T1}, \n C_T2 = {C_T2}")
        # print(f"\ndet(C_T1) = {det(C_T1)}, \n det(C_T2) = {det(C_T2)}")

        if matrix_is_negative(V):
            F_snmf, G_snmf, snmf_frobenius_error = \
                snmf_multiple_inits(V, number_initializations)
            # print(V_T1, "\n", Q, "\n", U, det(Q))
            # print(f"Q = {Q}, \ndet(Q) = {det(Q)}")
            M_possibly_not_ortho = G_snmf
            # import matplotlib.pyplot as plt
            print("\nsnmf_ferr = ", snmf_frobenius_error)
            # plt.matshow(M_possibly_not_ortho, aspect="auto")
            # plt.colorbar()
            # plt.show()
        else:
            M_possibly_not_ortho = V
            # F_snmf = None
            snmf_frobenius_error = None
    # if False:
    #     blah = True

    else:
        # Then it is a one target procedure
        if matrix_is_negative(V_T1):
            F_snmf, G_snmf, snmf_frobenius_error = \
                snmf_multiple_inits(V_T1, number_initializations)
            # print(V_T1, "\n", Q, "\n", U, det(Q))
            # print(f"Q = {Q}, \ndet(Q) = {det(Q)}")
            # print(np.linalg.norm(V_T1 - F_snmf@G_snmf))
            print("\nsnmf_ferr = ", snmf_frobenius_error)
            M_possibly_not_ortho = G_snmf
        else:
            M_possibly_not_ortho = V_T1
            # F_snmf = None
            snmf_frobenius_error = None

    # if matrix_is_negative(M_possibly_not_ortho):
    #     raise ValueError('The function '
    #                      'get_non_normalized_positive_reduction_matrix'
    #                      'does not reach its goal.'
    #                      ' There is probably an error in the function.')
    # if matrix_is_negative(M_possibly_not_ortho):
    if not matrix_is_orthogonal(M_possibly_not_ortho):
        # If the matrix is not already orthogonal...
        F_onmf, M_not_normalized, onmf_frobenius_error, onmf_ortho_error =\
            onmf_multiple_inits(M_possibly_not_ortho,
                                number_initializations)
        # import matplotlib.pyplot as plt
        print(f"\nonmf_ferr = {onmf_frobenius_error} ",
              f"\nonmf_oerr = {onmf_ortho_error}")
        # plt.matshow(M_possibly_not_ortho, aspect="auto")
        # plt.colorbar()
        # plt.show()
        # if not np.all(V_T2 == np.zeros((n, N))):
        #     if matrix_is_negative(V):
        #         print("||V - F_snmf@F_onmf@M_not_normalized|| = ",
        #               np.linalg.norm(V - F_snmf@F_onmf@M_not_normalized))
        # else:
        #     if matrix_is_negative(V_T1):
        #         print("||V - F_snmf@F_onmf@M_not_normalized|| = ",
        #               np.linalg.norm(V_T1
        #                              - F_snmf@F_onmf@M_not_normalized))

    else:
        M_not_normalized = M_possibly_not_ortho
        onmf_frobenius_error, onmf_ortho_error = None, None

    M = normalize_rows_matrix_M1(M_not_normalized)

    if not matrix_is_positive(M):
        raise ValueError("The reduced matrix M is not positive anymore after"
                         "using orthonormal matrix factorization.")

    return M, snmf_frobenius_error, onmf_frobenius_error, onmf_ortho_error