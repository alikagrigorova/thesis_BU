"""Utilities for NIG variational inference with beta-divergence and overlap diagnostics.

This module is extracted from notebook workflows so experiments can import stable,
reusable functions instead of duplicating notebook cells.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import beta as beta_fn
from scipy.special import gammaln, logsumexp
from scipy.stats import gaussian_kde, norm

from sklearn.metrics.pairwise import pairwise_kernels

import torch
import torch.nn as nn


Array = np.ndarray
NIGParams = Dict[str, object]


# ==============================
# Variational NIG core
# ==============================


class VariationalNIG(nn.Module):
    """Variational Normal-Inverse-Gamma family for Gaussian linear regression."""

    def __init__(self, p: int, init_V: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.p = p
        self.mu = nn.Parameter(torch.zeros(p, dtype=torch.float64))

        if init_V is None:
            init_V = torch.eye(p, dtype=torch.float64)
        else:
            init_V = torch.as_tensor(init_V, dtype=torch.float64)

        L_init = torch.linalg.cholesky(init_V)
        self.L_diag_log = nn.Parameter(torch.log(torch.diag(L_init)))
        self.L_offdiag = nn.Parameter(torch.tril(L_init, diagonal=-1))
        self.d_log = nn.Parameter(torch.log(torch.tensor(3.0, dtype=torch.float64)))
        self.beta_log = nn.Parameter(torch.log(torch.tensor(1.0, dtype=torch.float64)))

    @property
    def d(self) -> torch.Tensor:
        return torch.exp(self.d_log)

    @property
    def beta_param(self) -> torch.Tensor:
        return torch.exp(self.beta_log)

    @property
    def L(self) -> torch.Tensor:
        return self.L_offdiag + torch.diag(torch.exp(self.L_diag_log))

    @property
    def V(self) -> torch.Tensor:
        return self.L @ self.L.T

    def initialize_at_prior(
        self,
        mu_0: torch.Tensor,
        V_0: torch.Tensor,
        d_0: torch.Tensor,
        beta_0: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            self.mu.copy_(torch.as_tensor(mu_0, dtype=torch.float64))
            L_prior = torch.linalg.cholesky(torch.as_tensor(V_0, dtype=torch.float64))
            self.L_diag_log.copy_(torch.log(torch.diag(L_prior)))
            self.L_offdiag.copy_(torch.tril(L_prior, diagonal=-1))
            self.d_log.copy_(torch.log(torch.as_tensor(d_0, dtype=torch.float64)).reshape(()))
            self.beta_log.copy_(torch.log(torch.as_tensor(beta_0, dtype=torch.float64)).reshape(()))

    def initialize_from_params(self, params: NIGParams) -> None:
        self.initialize_at_prior(params["mu"], params["V"], params["d"], params["beta"])

    def get_params_dict(self) -> NIGParams:
        return {
            "mu": self.mu.detach(),
            "V": self.V.detach(),
            "d": float(self.d.detach().item()),
            "beta": float(self.beta_param.detach().item()),
        }


# ==============================
# NIG math + sampling
# ==============================


def sample_from_variational_nig(params: NIGParams, n_samples: int = 5000, seed: Optional[int] = None) -> Array:
    """Sample joint (sigma2, beta) from NIG parameters."""
    mu = np.asarray(params["mu"], dtype=float)
    V = np.asarray(params["V"], dtype=float)
    d = float(params["d"])
    beta_param = float(params["beta"])

    rng = np.random.default_rng(seed)
    p = len(mu)

    sigma2_samples = stats.invgamma.rvs(d, scale=beta_param, size=n_samples, random_state=rng)

    beta_samples = np.zeros((n_samples, p), dtype=float)
    for i in range(n_samples):
        cov = sigma2_samples[i] * V
        try:
            beta_samples[i] = stats.multivariate_normal.rvs(mu, cov, random_state=rng)
        except ValueError:
            beta_samples[i] = stats.multivariate_normal.rvs(mu, sigma2_samples[i] * np.eye(p), random_state=rng)

    return np.column_stack([sigma2_samples, beta_samples])


def standard_nig_posterior_torch(
    X: Array,
    y: Array,
    a0: float,
    b0: float,
    sigma0: float,
) -> NIGParams:
    X_t = torch.as_tensor(X, dtype=torch.float64)
    y_t = torch.as_tensor(y, dtype=torch.float64)
    n, p = X_t.shape

    V0_inv = (1.0 / (sigma0**2)) * torch.eye(p, dtype=torch.float64)
    Vn_inv = V0_inv + X_t.T @ X_t
    Vn = torch.linalg.inv(Vn_inv)
    mu_n = Vn @ (X_t.T @ y_t)
    a_n = a0 + 0.5 * n
    b_n = b0 + 0.5 * (y_t @ y_t - mu_n @ Vn_inv @ mu_n)

    return {
        "mu": mu_n.detach(),
        "V": Vn.detach(),
        "d": float(a_n),
        "beta": float(b_n),
    }


def kl_nig(
    q_mu: torch.Tensor,
    q_V: torch.Tensor,
    q_d: torch.Tensor,
    q_beta: torch.Tensor,
    p_mu: torch.Tensor,
    p_V: torch.Tensor,
    p_d: torch.Tensor,
    p_beta: torch.Tensor,
) -> torch.Tensor:
    """KL(q || p) for NIG distributions."""
    lgamma_p_d = torch.lgamma(p_d)
    lgamma_q_d = torch.lgamma(q_d)

    kl_sigma = (
        (q_d - p_d) * torch.digamma(q_d)
        + torch.log(q_beta ** (p_d + 1))
        + lgamma_p_d
        - torch.log(q_beta)
        - p_d * torch.log(p_beta)
        - lgamma_q_d
        + p_beta * (q_d / q_beta)
        - q_d
    )

    p_V_inv = torch.linalg.inv(p_V)
    diff = q_mu - p_mu

    kl_beta = 0.5 * (
        torch.trace(p_V_inv @ q_V)
        + diff @ p_V_inv @ diff
        - q_mu.shape[0]
        + torch.logdet(p_V)
        - torch.logdet(q_V)
    )

    return kl_sigma + kl_beta


def expected_beta_loss(
    X: Array,
    y: Array,
    q_mu: torch.Tensor,
    q_V: torch.Tensor,
    q_d: torch.Tensor,
    q_beta: torch.Tensor,
    beta_div: float,
) -> torch.Tensor:
    """Compute E_q[beta-divergence loss] for Gaussian linear regression."""
    b = torch.as_tensor(beta_div, dtype=torch.float64)
    a = b - 1.0

    X_t = torch.as_tensor(X, dtype=torch.float64)
    y_t = torch.as_tensor(y, dtype=torch.float64)

    n = X_t.shape[0]
    log_2pi = torch.log(torch.tensor(2.0 * np.pi, dtype=torch.float64))
    log_q_beta = torch.log(q_beta)

    residual_mean = y_t - X_t @ q_mu
    xVx = torch.sum((X_t @ q_V) * X_t, dim=1)

    eps_beta = torch.tensor(1e-8, dtype=torch.float64)
    if torch.abs(a) < eps_beta:
        term1 = 0.5 * torch.sum(
            log_2pi + log_q_beta - torch.digamma(q_d) + (residual_mean**2 + xVx) * q_d / q_beta
        )
    else:
        gamma_log_ratio = torch.lgamma(q_d + 0.5 * a) - torch.lgamma(q_d)
        power = q_d + 0.5 * a
        infl = torch.clamp(1.0 + a * xVx, min=1e-8)
        c = 0.5 * a * residual_mean**2 / infl

        log_a = (
            -0.5 * a * log_2pi
            - 0.5 * torch.log(infl)
            + q_d * log_q_beta
            + gamma_log_ratio
            - power * (log_q_beta + torch.log1p(c / q_beta))
        )
        term1 = torch.sum(-torch.expm1(log_a) / a)

    alpha = 0.5 * a
    log_moment_ig = -alpha * log_q_beta + torch.lgamma(q_d + alpha) - torch.lgamma(q_d)
    log_const2 = -torch.log(b) - alpha * log_2pi
    expected_term2 = n * torch.exp(log_const2 + log_moment_ig)

    return term1 + expected_term2


def make_beta_continuation_grid(beta_target: float, beta_anchor: float = 1.0 + 1e-6, max_step: float = 5e-4) -> List[float]:
    """Monotone beta grid from near-1 anchor to target with bounded steps."""
    beta_target = float(beta_target)
    beta_anchor = float(beta_anchor)
    if beta_target <= beta_anchor:
        return [beta_target]
    n_steps = max(2, int(np.ceil((beta_target - beta_anchor) / max_step)) + 1)
    return np.linspace(beta_anchor, beta_target, n_steps).tolist()


# ==============================
# Beta-GVI fitting
# ==============================


def gvi_beta_posterior(
    X: Array,
    y: Array,
    beta_div: float,
    a0: float,
    b0: float,
    sigma0: float,
    n_epochs: int = 1000,
    lr: float = 0.01,
    verbose: bool = False,
    init_params: Optional[NIGParams] = None,
) -> NIGParams:
    X_t = torch.as_tensor(X, dtype=torch.float64)
    y_t = torch.as_tensor(y, dtype=torch.float64)
    _, p = X_t.shape

    mu_0 = torch.zeros(p, dtype=torch.float64)
    V_0 = (sigma0**2) * torch.eye(p, dtype=torch.float64)
    d_0 = torch.tensor(a0, dtype=torch.float64)
    beta_0 = torch.tensor(b0, dtype=torch.float64)

    q = VariationalNIG(p)
    if init_params is None:
        q.initialize_at_prior(mu_0, V_0, d_0, beta_0)
    else:
        q.initialize_from_params(init_params)

    optimizer = torch.optim.Adam(q.parameters(), lr=lr)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        data_term = expected_beta_loss(X_t, y_t, q.mu, q.V, q.d, q.beta_param, beta_div)
        kl_term = kl_nig(q.mu, q.V, q.d, q.beta_param, mu_0, V_0, d_0, beta_0)
        loss = data_term + kl_term
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, loss: {loss.item():.4f}")

    return q.get_params_dict()


def gvi_beta_posterior_continuation(
    X: Array,
    y: Array,
    beta_div: float,
    a0: float,
    b0: float,
    sigma0: float,
    n_epochs: int = 300,
    lr: float = 0.005,
    verbose: bool = False,
    beta_anchor: float = 1.0 + 1e-6,
    max_step: float = 5e-4,
    init_params: Optional[NIGParams] = None,
    return_path: bool = False,
) -> NIGParams | Dict[float, NIGParams]:
    """Continuation fit: each beta initialized from previous fitted solution."""
    beta_grid = make_beta_continuation_grid(beta_div, beta_anchor=beta_anchor, max_step=max_step)
    path_results: Dict[float, NIGParams] = {}

    current_params = standard_nig_posterior_torch(X, y, a0=a0, b0=b0, sigma0=sigma0) if init_params is None else init_params

    for beta_val in beta_grid:
        current_params = gvi_beta_posterior(
            X,
            y,
            beta_div=beta_val,
            a0=a0,
            b0=b0,
            sigma0=sigma0,
            n_epochs=n_epochs,
            lr=lr,
            verbose=verbose,
            init_params=current_params,
        )
        path_results[round(beta_val, 6)] = current_params

    if return_path:
        return path_results
    return current_params


# ==============================
# Overlap helpers
# ==============================


def fit_standard_nig_posterior(X: Array, y: Array, a0: float, b0: float, sigma0: float) -> NIGParams:
    X = np.asarray(X)
    y = np.asarray(y)
    n, p = X.shape

    mu0 = np.zeros(p)
    V0 = (sigma0**2) * np.eye(p)
    V0_inv = np.linalg.inv(V0)

    Vn_inv = V0_inv + X.T @ X
    Vn = np.linalg.inv(Vn_inv)
    mu_n = Vn @ (V0_inv @ mu0 + X.T @ y)

    a_n = a0 + 0.5 * n
    b_n = b0 + 0.5 * (y @ y + mu0 @ V0_inv @ mu0 - mu_n @ Vn_inv @ mu_n)

    return {"mu": mu_n, "V": Vn, "d": float(a_n), "beta": float(b_n)}


def _stable_cholesky(matrix: Array, jitter: float = 1e-10, max_tries: int = 8) -> Tuple[Array, Array]:
    matrix = np.asarray(matrix, dtype=float)
    eye = np.eye(matrix.shape[0])
    current = matrix.copy()
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(current), current
        except np.linalg.LinAlgError:
            current = current + jitter * eye
            jitter *= 10.0
    raise np.linalg.LinAlgError("Unable to compute a stable Cholesky factor")


def nig_logpdf(theta: Array, params: NIGParams) -> Array | float:
    theta = np.asarray(theta, dtype=float)
    single_input = theta.ndim == 1
    if single_input:
        theta = theta[None, :]

    sigma2 = theta[:, 0]
    beta_vec = theta[:, 1:]

    mu = np.asarray(params["mu"], dtype=float)
    V = np.asarray(params["V"], dtype=float)
    d = float(params["d"])
    b = float(params["beta"])

    _, V_stable = _stable_cholesky(V)
    sign, logdetV = np.linalg.slogdet(V_stable)
    out = np.full(theta.shape[0], -np.inf)
    if sign <= 0:
        return float(out[0]) if single_input else out

    V_inv = np.linalg.inv(V_stable)
    diff = beta_vec - mu
    quad = np.einsum("ni,ij,nj->n", diff, V_inv, diff)

    positive = sigma2 > 0.0
    if np.any(positive):
        sigma2_pos = sigma2[positive]
        quad_pos = quad[positive]
        p = len(mu)
        log_p_sigma = d * np.log(b) - gammaln(d) - (d + 1.0) * np.log(sigma2_pos) - b / sigma2_pos
        log_p_beta_given_sigma = -0.5 * (
            p * np.log(2.0 * np.pi) + logdetV + p * np.log(sigma2_pos) + quad_pos / sigma2_pos
        )
        out[positive] = log_p_sigma + log_p_beta_given_sigma

    return float(out[0]) if single_input else out


def sample_from_nig_params(params: NIGParams, n_samples: int = 10000, seed: Optional[int] = None) -> Array:
    mu = np.asarray(params["mu"], dtype=float)
    V = np.asarray(params["V"], dtype=float)
    d = float(params["d"])
    b = float(params["beta"])

    rng = np.random.default_rng(seed)
    p = len(mu)
    sigma2 = stats.invgamma.rvs(d, scale=b, size=n_samples, random_state=rng)

    L, _ = _stable_cholesky(V)
    z = rng.standard_normal((n_samples, p))
    scaled = z @ L.T
    beta_samples = mu + np.sqrt(sigma2)[:, None] * scaled

    return np.column_stack([sigma2, beta_samples])


def bhattacharyya_nig_closed_form(params1: NIGParams, params2: NIGParams) -> float:
    """Closed-form Bhattacharyya coefficient for two NIG posteriors.

    The coefficient is

        BC(p1, p2) = ∫ sqrt(p1(theta) p2(theta)) dtheta,

    where theta = (sigma2, beta) and both p1, p2 are Normal-Inverse-Gamma.
    """
    mu1 = np.asarray(params1["mu"], dtype=float)
    mu2 = np.asarray(params2["mu"], dtype=float)
    V1 = np.asarray(params1["V"], dtype=float)
    V2 = np.asarray(params2["V"], dtype=float)
    d1 = float(params1["d"])
    d2 = float(params2["d"])
    b1 = float(params1["beta"])
    b2 = float(params2["beta"])

    if b1 <= 0.0 or b2 <= 0.0:
        raise ValueError("NIG beta parameters must be positive")

    p = mu1.shape[0]
    _, V1_stable = _stable_cholesky(V1)
    _, V2_stable = _stable_cholesky(V2)

    sign1, logdetV1 = np.linalg.slogdet(V1_stable)
    sign2, logdetV2 = np.linalg.slogdet(V2_stable)
    if sign1 <= 0 or sign2 <= 0:
        raise ValueError("NIG covariance matrices must be positive definite")

    A1 = np.linalg.inv(V1_stable)
    A2 = np.linalg.inv(V2_stable)
    S = A1 + A2
    _, S_stable = _stable_cholesky(S)
    signS, logdetS = np.linalg.slogdet(S_stable)
    if signS <= 0:
        raise ValueError("Combined precision matrix must be positive definite")

    h_vec = A1 @ mu1 + A2 @ mu2
    m_vec = np.linalg.solve(S_stable, h_vec)
    quad_const = (
        mu1 @ A1 @ mu1
        + mu2 @ A2 @ mu2
        - h_vec @ m_vec
    )
    lam = 0.5 * (b1 + b2) + 0.25 * quad_const
    d_bar = 0.5 * (d1 + d2)

    log_bc = (
        0.5 * (d1 * np.log(b1) + d2 * np.log(b2))
        - 0.5 * (gammaln(d1) + gammaln(d2))
        - 0.25 * (logdetV1 + logdetV2)
        + 0.5 * p * np.log(2.0)
        - 0.5 * logdetS
        + gammaln(d_bar)
        - d_bar * np.log(lam)
    )
    return float(np.clip(np.exp(log_bc), 0.0, 1.0))


def hellinger_nig_closed_form(params1: NIGParams, params2: NIGParams) -> Tuple[float, float]:
    """Closed-form d_1/2 and Bhattacharyya coefficient for NIG posteriors."""
    bc = bhattacharyya_nig_closed_form(params1, params2)
    d12 = float(-2.0 * np.log(np.clip(bc, 1e-300, 1.0)))
    return d12, bc


def make_k_blocks(X: Array, y: Array, K: int = 8, seed: int = 123) -> List[Tuple[Array, Array]]:
    """Randomly split a dataset (X, y) into K approximately equal blocks."""
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    block_idx = np.array_split(perm, K)
    return [(X[idx], y[idx]) for idx in block_idx]


def u_statistic_hellinger_within(params_list: Sequence[NIGParams], n_is: int = 4000) -> Tuple[float, Array]:
    K = len(params_list)
    if K < 2:
        raise ValueError("Need at least K=2 blocks")

    pair_vals: List[float] = []
    for k in range(K):
        for l in range(k + 1, K):
            h_kl, _ = hellinger_nig_closed_form(params_list[k], params_list[l])
            pair_vals.append(h_kl)

    pair_arr = np.asarray(pair_vals)
    U = float((2.0 / (K * (K - 1))) * np.sum(pair_arr))
    return U, pair_arr


# ==============================
# True-DGP threshold helpers
# ==============================


# ==============================
# Plotting helpers
# ==============================


def plot_standard_posterior_overlap(
    beta_samples1: Array,
    beta_samples2: Array,
    beta_true: Optional[Array] = None,
    n_obs: Optional[int] = None,
    figsize: Tuple[int, int] = (15, 8),
    suptitle: str = "Posterior Distributions for Beta Coefficients (Misspecified Linear Model)",
) -> Tuple[plt.Figure, Array, Array]:
    """Plot histogram + KDE overlap panels and return (fig, axes, d_1/2 values)."""
    b1 = np.asarray(beta_samples1)
    b2 = np.asarray(beta_samples2)
    D = b1.shape[1]

    fig, axes = plt.subplots(2, D, figsize=figsize, squeeze=False)
    fig.suptitle(suptitle, fontsize=16)

    hellinger_distances: List[float] = []

    for i in range(D):
        label_1 = "Dataset 1" if n_obs is None else f"Dataset 1 (N={n_obs})"
        label_2 = "Dataset 2" if n_obs is None else f"Dataset 2 (N={n_obs})"

        axes[0, i].hist(b1[:, i], bins=50, alpha=0.7, density=True, color="blue", label=label_1)
        axes[0, i].axvline(np.mean(b1[:, i]), color="blue", linestyle="--", label=f"Mean: {np.mean(b1[:, i]):.3f}")

        axes[0, i].hist(b2[:, i], bins=50, alpha=0.7, density=True, color="red", label=label_2)
        axes[0, i].axvline(np.mean(b2[:, i]), color="red", linestyle="--", label=f"Mean: {np.mean(b2[:, i]):.3f}")

        if beta_true is not None:
            axes[0, i].axvline(beta_true[i], color="black", linewidth=2, label=f"True: {beta_true[i]:.3f}")

        axes[0, i].set_title(f"Beta {i + 1}")
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)

        kde1 = gaussian_kde(b1[:, i])
        kde2 = gaussian_kde(b2[:, i])
        combined = np.concatenate([b1[:, i], b2[:, i]])
        x_grid = np.linspace(np.percentile(combined, 1), np.percentile(combined, 99), 200)
        d1 = kde1(x_grid)
        d2 = kde2(x_grid)

        affinity = np.trapz(np.sqrt(d1 * d2), x_grid)
        h = float(-2.0 * np.log(np.clip(affinity, 1e-300, 1.0)))
        hellinger_distances.append(h)

        axes[1, i].plot(x_grid, d1, color="blue", linewidth=2, label="Dataset 1")
        axes[1, i].plot(x_grid, d2, color="red", linewidth=2, label="Dataset 2")
        axes[1, i].fill_between(x_grid, np.minimum(d1, d2), alpha=0.3, color="gray", label="Overlap region")
        axes[1, i].set_title(f"KDE Overlap (d_1/2={h:.3f})")
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, axes, np.asarray(hellinger_distances)


def plot_beta_posterior_overlap(
    beta_samples1: Array,
    beta_samples2: Array,
    beta_true: Optional[Array] = None,
    title: str = "Posterior Overlap for Beta Coefficients",
) -> Tuple[plt.Figure, Array]:
    """Plot posterior density overlap per coefficient and return (fig, axes)."""
    b1 = np.asarray(beta_samples1)
    b2 = np.asarray(beta_samples2)
    D = b1.shape[1]

    fig, axes = plt.subplots(1, D, figsize=(5 * D, 4))
    if D == 1:
        axes = np.asarray([axes])

    for i in range(D):
        kde1 = gaussian_kde(b1[:, i])
        kde2 = gaussian_kde(b2[:, i])

        combined = np.concatenate([b1[:, i], b2[:, i]])
        x_grid = np.linspace(np.percentile(combined, 1), np.percentile(combined, 99), 250)
        d1 = kde1(x_grid)
        d2 = kde2(x_grid)

        axes[i].plot(x_grid, d1, color="tab:blue", label="Posterior 1")
        axes[i].plot(x_grid, d2, color="tab:red", label="Posterior 2")
        axes[i].fill_between(x_grid, np.minimum(d1, d2), alpha=0.25, color="gray", label="Overlap")

        if beta_true is not None:
            axes[i].axvline(beta_true[i], color="black", linestyle="--", label="True")

        affinity = np.trapz(np.sqrt(d1 * d2), x_grid)
        h = float(-2.0 * np.log(np.clip(affinity, 1e-300, 1.0)))
        axes[i].set_title(f"beta[{i}] | d_1/2={h:.3f}")
        axes[i].grid(alpha=0.3)

    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_sigma2_overlap(
    sigma2_samples1: Array,
    sigma2_samples2: Array,
    title: str = "Sigma2 Posterior Overlap",
) -> Tuple[plt.Figure, plt.Axes]:
    s1 = np.asarray(sigma2_samples1)
    s2 = np.asarray(sigma2_samples2)

    fig, ax = plt.subplots(figsize=(7, 4))
    kde1 = gaussian_kde(s1)
    kde2 = gaussian_kde(s2)
    combined = np.concatenate([s1, s2])
    x_grid = np.linspace(np.percentile(combined, 1), np.percentile(combined, 99), 250)
    d1 = kde1(x_grid)
    d2 = kde2(x_grid)

    ax.plot(x_grid, d1, color="tab:blue", label="Posterior 1")
    ax.plot(x_grid, d2, color="tab:red", label="Posterior 2")
    ax.fill_between(x_grid, np.minimum(d1, d2), alpha=0.25, color="gray", label="Overlap")

    affinity = np.trapz(np.sqrt(d1 * d2), x_grid)
    h = float(-2.0 * np.log(np.clip(affinity, 1e-300, 1.0)))
    ax.set_title(f"{title} | d_1/2={h:.3f}")
    ax.set_xlabel("sigma2")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig, ax


# ==============================
# 1D location-model calibration helpers
# ==============================


# ==============================
# Posterior predictive log-score
# ==============================


def nig_predictive_logpdf(
    params: NIGParams,
    X_test: Array,
    y_test: Array,
) -> float:
    """Mean log predictive density under a NIG posterior (closed-form Student-t).

    For NIG(mu, V, d, beta) with Gaussian likelihood, the predictive for a new
    observation y* given x* is:

        y* ~ t_{2d}(x*' mu, (beta/d)(1 + x*' V x*))

    Returns the average log p(y_i | x_i, posterior) over all test points.
    """
    mu = np.asarray(params["mu"], dtype=float)
    V = np.asarray(params["V"], dtype=float)
    d = float(params["d"])
    b = float(params["beta"])

    X_test = np.asarray(X_test, dtype=float)
    y_test = np.asarray(y_test, dtype=float).ravel()

    df = 2.0 * d
    loc = X_test @ mu                          # (n_test,)
    xVx = np.sum((X_test @ V) * X_test, axis=1)  # (n_test,)
    scale = np.sqrt((b / d) * (1.0 + xVx))    # (n_test,)

    lp = stats.t.logpdf(y_test, df=df, loc=loc, scale=scale)
    return float(np.mean(lp))


def gibbs_predictive_logpdf(
    beta_samples: Array,
    sigma2_samples: Array,
    X_test: Array,
    y_test: Array,
    nu: float,
) -> float:
    """Mean log predictive density under a Student-t Gibbs posterior (MC estimate).

    For each posterior draw (beta^(s), sigma2^(s)), the observation model is

        y | x, beta, sigma2 ~ t_nu(x' beta, sigma2)

    so

        log p(y_i | D) ≈ log(1/S sum_s t_nu(y_i; x_i' beta^(s), sigma2^(s)))

    Uses log-sum-exp for numerical stability.

    Returns the average log-predictive across test points.
    """
    beta_samples = np.asarray(beta_samples, dtype=float)
    sigma2_samples = np.asarray(sigma2_samples, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_test = np.asarray(y_test, dtype=float).ravel()

    S = beta_samples.shape[0]
    n_test = len(y_test)

    # loc_mat: (S, n_test)
    loc_mat = beta_samples @ X_test.T
    scale_mat = np.sqrt(sigma2_samples)[:, None]  # (S, 1) broadcast

    # log p(y_i | beta^(s), sigma2^(s))  shape (S, n_test)
    log_lik = stats.t.logpdf(y_test[None, :], df=nu, loc=loc_mat, scale=scale_mat)

    # log-mean-exp across S for each test point
    log_pred = logsumexp(log_lik, axis=0) - np.log(S)  # (n_test,)
    return float(np.mean(log_pred))


# ---------------------------------------------------------------------------
# Tsallis scoring rule
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Rényi scoring rule
# ---------------------------------------------------------------------------


# ==============================
# 1D Gaussian location-model helpers
# ==============================


def standard_posterior_params(y: Array, mu0: float, tau2: float, sigma2: float) -> Tuple[float, float]:
    """Closed-form Gaussian posterior mean/variance for a location model."""
    n = len(y)
    ybar = np.mean(y)
    post_var = 1 / (n / sigma2 + 1 / tau2)
    post_mean = post_var * (n * ybar / sigma2 + mu0 / tau2)
    return post_mean, post_var


def standard_bayes_posterior(y: Array, theta_grid: Array, mu0: float, tau2: float, sigma2: float) -> Array:
    """Closed-form Gaussian posterior density for a location model, evaluated on a grid."""
    post_mean, post_var = standard_posterior_params(y, mu0, tau2, sigma2)
    return norm.pdf(theta_grid, post_mean, np.sqrt(post_var))


def beta_bayes_posterior_grid(y: Array, theta_grid: Array, beta: float, sigma2: float, mu0: float, tau2: float) -> Array:
    """Beta-Bayes posterior density for a Gaussian location model, evaluated on a grid (no closed form)."""
    log_prior = np.log(norm.pdf(theta_grid, mu0, np.sqrt(tau2)))
    const = (2 * np.pi * sigma2) ** (-(beta - 1) / 2)
    beta_like = np.zeros_like(theta_grid)
    for yi in y:
        beta_like += const * np.exp(
            - (beta - 1) * (yi - theta_grid) ** 2 / (2 * sigma2)
        )
    log_post = log_prior + beta_like / (beta - 1)
    log_post -= np.max(log_post)
    post = np.exp(log_post)
    post /= np.trapz(post, theta_grid)
    return post


def bhattacharyya_1d_gaussian(mu1: float, sigma1_sq: float, mu2: float, sigma2_sq: float) -> float:
    """Closed-form Bhattacharyya coefficient for two 1D Gaussians."""
    sigma1 = np.sqrt(sigma1_sq)
    sigma2 = np.sqrt(sigma2_sq)
    denom = sigma1_sq + sigma2_sq

    bc = np.sqrt(2 * sigma1 * sigma2 / denom) * np.exp(
        -0.25 * (mu1 - mu2) ** 2 / denom
    )
    return bc


def hellinger_1d_gaussian(mu1: float, sigma1_sq: float, mu2: float, sigma2_sq: float) -> float:
    """Closed-form d_1/2 for two 1D Gaussians."""
    bc = bhattacharyya_1d_gaussian(mu1, sigma1_sq, mu2, sigma2_sq)
    return -2.0 * np.log(np.clip(bc, 1e-300, 1.0))


# ==============================
# Joint (multivariate) predictive discrepancy over a fixed test set
# ==============================


def nig_joint_predictive(params: NIGParams, X_star: Array) -> Tuple[Array, Array]:
    """Joint multivariate Gaussian predictive over X_star.

    Returns (mu_pred, Sigma_pred) where Sigma_pred is (n x n).
    y* | X*, D ~ N(X* mu, (b/d)(I + X* V X*'))
    """
    mu = np.asarray(params["mu"], dtype=float)
    V = np.asarray(params["V"], dtype=float)
    d = float(params["d"])
    b = float(params["beta"])
    n = X_star.shape[0]
    mu_pred = X_star @ mu
    Sigma_pred = (b / d) * (np.eye(n) + X_star @ V @ X_star.T)
    return mu_pred, Sigma_pred


def minus2logBC_mvn(mu1: Array, Sigma1: Array, mu2: Array, Sigma2: Array) -> float:
    """Joint -2 log Bhattacharyya coefficient between two multivariate Gaussians.

    = 0.25 * (mu1-mu2)' Sigma_bar^{-1} (mu1-mu2) + log|Sigma_bar| - 0.5*(log|Sigma1| + log|Sigma2|)
    Returns a single scalar.
    """
    Sigma_bar = 0.5 * (Sigma1 + Sigma2)
    delta = mu1 - mu2
    _, ld1 = np.linalg.slogdet(Sigma1)
    _, ld2 = np.linalg.slogdet(Sigma2)
    _, ld_bar = np.linalg.slogdet(Sigma_bar)
    maha = float(delta @ np.linalg.solve(Sigma_bar, delta))
    return 0.25 * maha + ld_bar - 0.5 * (ld1 + ld2)


# ==============================
# Leave-one-block-out predictive metrics (Student-t NIG predictive)
# ==============================


def nig_predictive_params(params: NIGParams, X_test: Array) -> Tuple[Array, Array, Array]:
    """Return (df, loc, scale) arrays for the marginal Student-t predictive at each test row."""
    mu = np.asarray(params["mu"], dtype=float)
    V = np.asarray(params["V"], dtype=float)
    d = float(params["d"])
    b = float(params["beta"])
    X_test = np.asarray(X_test, dtype=float)

    df = 2.0 * d
    loc = X_test @ mu
    xVx = np.sum((X_test @ V) * X_test, axis=1)
    scale = np.sqrt((b / d) * (1.0 + xVx))
    return df, loc, scale


def predictive_coverage(df: float, loc: Array, scale: Array, y_test: Array, level: float = 0.95) -> float:
    """Fraction of y_test inside the level% prediction interval."""
    alpha = (1.0 - level) / 2.0
    lo = stats.t.ppf(alpha, df=df, loc=loc, scale=scale)
    hi = stats.t.ppf(1 - alpha, df=df, loc=loc, scale=scale)
    return float(np.mean((y_test >= lo) & (y_test <= hi)))


def predictive_mlpd(df: float, loc: Array, scale: Array, y_test: Array) -> float:
    """Mean log-predictive density under the Student-t predictive."""
    return float(np.mean(stats.t.logpdf(y_test, df=df, loc=loc, scale=scale)))


def predictive_crps_student(df: float, loc: Array, scale: Array, y_test: Array) -> float:
    """Closed-form CRPS for a univariate Student-t predictive (requires df > 1).

    Following Gneiting & Raftery (2007) / Jordan, Krueger & Lerch (2019):
    CRPS(t_df, y) = scale * [ z(2F(z)-1) + 2f(z)(df+z^2)/(df-1)
                               - 2*sqrt(df)*B(1/2, df-1/2) / ((df-1)*B(1/2, df/2)^2) ]
    where z=(y-loc)/scale, f/F are the standard Student-t pdf/cdf.

    NOTE: an earlier version of this helper (matching what the source notebooks
    originally computed) used an incorrect spread constant -- verified numerically
    against a Monte Carlo CRPS estimator, that version overstates CRPS increasingly
    as df grows (by a factor of ~1.3 at df=20, ~2.7 at df=18000, the regime this
    module's leave-one-block-out CV predictives fall in since df=2*(a0+n/2) with
    n in the thousands). This corrected formula matches the Monte Carlo estimator
    to within simulation noise across the df range tested.
    """
    z = (y_test - loc) / scale
    pdf_z = stats.t.pdf(z, df=df)
    cdf_z = stats.t.cdf(z, df=df)
    spread_const = (
        2.0 * np.sqrt(df) * beta_fn(0.5, df - 0.5) / ((df - 1) * beta_fn(0.5, df / 2) ** 2)
    )
    crps_per_point = scale * (
        z * (2 * cdf_z - 1)
        + 2 * pdf_z * (df + z**2) / (df - 1)
        - spread_const
    )
    return float(np.mean(crps_per_point))


# ==============================
# Synthetic regression data generation and closed-form NIG linear regression
# (trimmed from the external Bag_code/ project so this module is self-contained)
# ==============================


def _nonzero_sparse_inds(k: int, D: int) -> Array:
    inds = np.array([(i * (D + 0.5)) // (k + 1) for i in range(1, k + 1)], dtype=int) - 1
    return inds


def _get_beta_true(sparsity: str, D: int) -> Array:
    """True coefficient vector for a given sparsity pattern (see generate_synthetic_data)."""
    if sparsity == "dense":
        return 2 ** (2 - np.arange(D) / 2)
    elif sparsity.startswith("denser"):
        if len(sparsity) == 6:
            return 4 / np.sqrt(1 + np.arange(D))
        else:
            power = float(sparsity[6:])
            return 4 / (1 + np.arange(D)) ** power
    elif sparsity[-6:] != "sparse":
        raise ValueError(f"invalid sparsity type {sparsity}")
    k = int(sparsity[:-6])
    beta_vec = np.zeros(D)
    inds = _nonzero_sparse_inds(k, D)
    beta_vec[inds] = 1
    return beta_vec


def _generate_y_for_X_beta(
    noisetype: str,
    X: Array,
    beta_vec: Array,
    error_vars_fun: Optional[Callable[[Array], Array]] = None,
) -> Array:
    means = X.dot(beta_vec)
    N = means.size
    err = np.random.randn(N)
    if error_vars_fun is not None:
        err = np.sqrt(error_vars_fun(X)) * err
    if noisetype == "gaussian":
        y = means + err
    elif noisetype == "heavy":
        # Standard normal / sqrt(chi-square(df)/df) ~ Student-t(df); df is fixed at 4
        # here (this predates and is independent of any t(nu) used elsewhere in this
        # module for GVI/Gibbs machinery -- kept as-is for parity with prior results).
        df = 4
        y = means + err / np.sqrt(np.random.chisquare(df, N) / df)
    else:
        raise ValueError(f"invalid noise type {noisetype}")
    return y


def generate_synthetic_data(
    mode: str,
    seed: int,
    D: int,
    N: int,
    X: Optional[Array] = None,
    error_vars_fun: Optional[Callable[[Array], Array]] = None,
) -> Tuple[Array, Array, Array]:
    """Generate a synthetic regression dataset for a mode string of the form
    ``synthetic-<corrtype>-<sparsity>-<regtype>-<noisetype>``.

    Note: for ``corrtype in {"uncorr", "fixed"}`` the design matrix X is drawn from an
    unseeded ``np.random.default_rng()``, so ``seed`` only controls the noise draw
    (via the legacy global ``np.random.seed``), not X itself. This matches the
    behavior of the original Bag_code implementation these MC replicates were
    generated with.
    """
    rng = np.random.default_rng()
    mode_parts = mode.split("-")
    if len(mode_parts) != 5:
        raise ValueError("invalid mode")
    _, corrtype, sparsity, regtype, noisetype = mode_parts
    np.random.seed(seed)
    if X is None:
        if corrtype.startswith("corrsimple"):
            stdev = 1 if len(corrtype) == 10 else float(corrtype[10:])
            scale = 8
            locs = np.linspace(0, D / scale, D, endpoint=False).reshape(-1, 1)
            cov = stdev**2 * pairwise_kernels(locs, metric="rbf")
            X = rng.multivariate_normal(np.zeros(cov.shape[0]), cov, N)
        elif corrtype.startswith("corr"):
            scale = 8 if len(corrtype) == 4 else float(corrtype[4:])
            df = 10
            locs = np.linspace(0, D / scale, D, endpoint=False).reshape(-1, 1)
            cov = pairwise_kernels(locs, metric="rbf") + 1e-10 * np.eye(D)
            rescale = np.ones((N, D))
            rescale[:, ::2] = 1 / np.sqrt(rng.chisquare(df, (N, 1)) / (df - 2))
            covs = np.einsum("ij,ki,kj->kij", cov, rescale, rescale)
            mvn = np.vectorize(
                lambda c: rng.multivariate_normal(np.zeros(c.shape[0]), c, method="cholesky"),
                signature="(n,n)->(n)",
            )
            X = mvn(covs)
        elif corrtype == "uncorr" or corrtype.startswith("fixed"):
            X = rng.normal(size=(N, D))
            if corrtype == "fixed":
                X[:, 0] = 1
        else:
            raise ValueError(f"invalid correlation type {corrtype}")
    beta0 = _get_beta_true(sparsity, D)
    if regtype == "nonlinear":
        Xgen = X**3
        beta_opt = 3 * beta0
    elif regtype == "linear":
        Xgen = X
        beta_opt = beta0
    else:
        raise ValueError(f"invalid regression type {regtype}")
    y = _generate_y_for_X_beta(noisetype, Xgen, beta0, error_vars_fun)
    return X, y, beta_opt


_sigma2_posterior_means_log: List[float] = []


def posterior_samples(n_samples: int, muN: Array, precN: Array, aN: float, bN: float) -> Array:
    """Draw joint (sigma2, coefficient) samples from a closed-form NIG posterior."""
    Sigma = bN / aN * np.linalg.inv(precN)
    samples = np.zeros((n_samples, muN.size + 1))
    samples[:, 0] = stats.invgamma.rvs(aN, scale=bN, size=n_samples)
    _sigma2_posterior_means_log.append(bN / (aN - 1))
    samples[:, 1:] = stats.multivariate_t.rvs(loc=muN, shape=Sigma, df=2 * aN, size=n_samples)
    return samples


def linreg(
    X: Array,
    y: Array,
    n_samples: int,
    beta_opt: Array,
    w: Optional[Array] = None,
    a0: float = 1.0,
    b0: float = 1.0,
    sigma0: float = 1.0,
    verbose: bool = False,
) -> Tuple[Array, Array]:
    """Closed-form NIG posterior samples for Bayesian linear regression.

    Prior: Normal-inverse-gamma(mu=0, lambda=1/sigma0^2, a0, b0).
    Returns (samples, opt_quantiles) where samples[:, 0] is sigma2 and
    samples[:, 1:] are the regression-coefficient draws.
    """
    if w is None:
        w = np.ones(X.shape[0])
    N = np.sum(w)
    Xw = np.sqrt(w[:, np.newaxis]) * X
    yw = np.sqrt(w) * y
    prec0 = 1 / sigma0**2
    precN = Xw.T.dot(Xw)
    precN[np.diag_indices_from(precN)] += prec0
    Xy = Xw.T.dot(yw)
    muN = np.linalg.solve(precN, Xy)
    aN = a0 + 0.5 * N
    bN = b0 + 0.5 * (yw.dot(yw) - muN.dot(Xy))
    if verbose:
        print(precN[0, 0], prec0, bN / (aN - 1))
    samples = posterior_samples(n_samples, muN, precN, aN, bN)
    opt_quantiles = np.mean(samples[:, 1:] < beta_opt[np.newaxis, :], axis=0)
    return samples, opt_quantiles
