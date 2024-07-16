import numpy as np
from numpy.linalg import solve, pinv
from sklearn.base import BaseEstimator
import time
from tqdm import tqdm


def get_norm(x, M=None):
    x2 = x * x if M is None else (x @ M) * x
    return np.sum(x2, axis=1, keepdims=True)


def euclidean_distances(samples, centers, M=None, squared=True, threshold=None):
    samples_norm = get_norm(samples, M)
    centers_norm = get_norm(centers, M) if samples is not centers else samples_norm
    distances = samples @ (M @ centers.T)
    distances = -2 * distances + samples_norm + centers_norm.T
    if threshold is not None:
        distances[distances < threshold] = 0
    if not squared:
        distances = np.sqrt(distances)

    return distances


def laplace_kernel(samples, centers, bandwidth, M=None):
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers, M=M, squared=False, threshold=0)
    kernel_mat = np.exp(-kernel_mat / bandwidth)
    return kernel_mat


def get_grads(X, sol, L, P, max_num_samples=20000):
    if len(X) > max_num_samples:
        indices = np.random.randint(len(X), size=max_num_samples)
        x = X[indices, :]
    else:
        x = X

    K = laplace_kernel(X, x, L, M=P)

    dist = euclidean_distances(X, x, M=P, squared=False, threshold=1e-10)

    with np.errstate(divide='ignore'):
        K = K / dist

    # RK: Why set to 0 and not some large number?
    K[K == float("Inf")] = 0.

    a1 = sol.T
    n, d = X.shape
    n, c = a1.shape
    m, d = x.shape

    a1 = a1.reshape(n, c, 1)

    X1 = (X @ P).reshape(n, 1, d)

    step1 = a1 @ X1
    del a1, X1
    step1 = step1.reshape(-1, c*d)

    step2 = K.T @ step1
    del step1

    step2 = step2.reshape(-1, c, d)

    a2 = sol
    step3 = (a2 @ K).T

    del K, a2

    step3 = step3.reshape(m, c, 1)
    x1 = (x @ P).reshape(m, 1, d)
    step3 = step3 @ x1

    G = (step2 - step3) * -1/L
    return G

def egop(G, verbose=False, diag_only=False):
    M = 0.
    chunks = len(G) // 20 + 1
    batches = np.array_split(G, chunks)
    i_range = range(len(batches))
    if verbose:
        i_range = tqdm(i_range)
    for i in i_range:
        grad = batches[i]
        gradT = np.swapaxes(grad, 1, 2)
        if diag_only:
            T = np.sum(gradT * gradT, axis=-1)
        else:
            T = gradT @ grad
        M += np.sum(T, axis=0)
        del grad, gradT
    M /= len(G)
    if diag_only:
        M = np.diag(M)
    return M


class RFM(BaseEstimator):
    def __init__(self, kernel="laplace"):
        self.kernel = kernel
        self.X_train = None
        self.alphas = None
        self.M = None
        self.L = None
        self.reg = None
        self.verbose = False

    def print_if_verbose(self, message):
        if self.verbose:
            print(message)

    def fit(self, X_train, y_train, reg=1e-3, bandwidth=10, num_iters=5,
            M=None, centering=True, verbose=False, diag_only=False,
            verify_gradients=False):
        self.X_train = X_train
        self.verbose = verbose
        n, d = X_train.shape
        if M is None:
            M = np.eye(d)
        self.M = M
        self.L = bandwidth
        self.reg = reg
        iter_idx = -1
        while True:
            iter_idx += 1
            self.print_if_verbose(f"Starting Iteration: {iter_idx}")
            start = time.time()
            K_train = laplace_kernel(X_train, X_train, self.L, M=M)
            sol = solve(K_train + reg * np.eye(n), y_train).T
            end = time.time()
            self.print_if_verbose(f"Solved Kernel Regression in {end - start} seconds.")
            self.alphas = sol
            if iter_idx == num_iters:
                break

            start = time.time()
            G = self.get_gradient_analytically(X_train)

            if verify_gradients:
                G_numerical = self.get_gradient_numerically(X_train)
                assert np.allclose(G, G_numerical, atol=1e-2)

            if centering:
                G -= np.mean(G, axis=0, keepdims=True)

            end = time.time()
            self.print_if_verbose(f"Computed Gradients in {end - start} seconds.")

            start = time.time()
            M = egop(G, verbose=verbose, diag_only=diag_only)
            end = time.time()
            self.print_if_verbose(f"Computed EGOP in {end - start}\n{'='*60}")
            self.M = M

        self.alphas = sol
        return self

    def get_gradient_numerically(self, X, eps=1e-1):
        d = X.shape[1]
        G_numeric = np.zeros((X.shape[0], self.alphas.shape[0], d))
        unperturbed_y = self.predict(X)
        for j in range(d):
            X_perturbed = X.copy()
            X_perturbed[:, j] += eps
            perturbed_y = self.predict(X_perturbed)
            G_numeric[:, :, j] = (perturbed_y - unperturbed_y) / eps
        return G_numeric

    def get_gradient_analytically(self, X):
        return get_grads(X, self.alphas, self.L, self.M)

    def predict(self, X_test):
        K_test = laplace_kernel(self.X_train, X_test, self.L, M=self.M)
        return (self.alphas @ K_test).T

    def get_alphas(self):
        return self.alphas

    def get_M(self):
        return self.M
