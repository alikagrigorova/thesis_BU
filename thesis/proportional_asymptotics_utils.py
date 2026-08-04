"""Utilities for the proportional-asymptotics beta* experiment: does beta*
(the beta value matching the observed discrepancy to the well-specified
threshold) stabilize as D grows at fixed kappa=D/N?

Model: Gaussian linear regression with an NIG(a0, b0, V0=sigma0^2*I) prior.
Misspecified DGP: t(5) noise (variance 5/3). Well-specified comparator: Gaussian
noise with the SAME variance (5/3), so the two are scale-matched.

Reuses the exact-NIG-posterior and closed-form NIG/Gaussian Bhattacharyya
machinery already in nig_beta_overlap_utils.py rather than duplicating it; only
adds what's new here: the t(5)/matched-Gaussian data generator, the plain
(non-variational) Gaussian beta-loss, and a JAX-based Laplace fitter (L-BFGS-B
MAP + exact Hessian) for the beta-posterior approximation.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize
import torch
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import nig_beta_overlap_utils as nbu

Array = np.ndarray

T5_VARIANCE = 5.0 / 3.0  # Var[t(5)] = nu/(nu-2) = 5/3


# ==============================
# Data generation
# ==============================

def generate_regression_data(
    D: int, N: int, seed: int, beta_true: Array, noise: str = "t5"
) -> Tuple[Array, Array]:
    """iid standard-normal design X (N x D); y = X @ beta_true + noise.

    noise="t5": Student-t(5) errors, scaled to unit variance then... no scaling
      needed -- t(5) already has the target variance 5/3 in its raw form, used
      as-is (matches the "misspecified" DGP).
    noise="gaussian_matched": N(0, 5/3) errors -- same variance as t(5), used
      for the well-specified threshold (Section 2), so the comparison is
      scale-matched rather than confounded by a variance mismatch.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, D))
    mean = X @ beta_true
    if noise == "t5":
        err = rng.standard_t(df=5, size=N)
    elif noise == "gaussian_matched":
        err = rng.normal(scale=np.sqrt(T5_VARIANCE), size=N)
    else:
        raise ValueError(f"invalid noise type {noise}")
    y = mean + err
    return X, y


def default_beta_true(D: int, seed: int = 0, scale: float = 1.0) -> Array:
    """Fixed, moderate-magnitude true coefficient vector, shared across all
    replications/datasets for a given D (only the data/noise varies)."""
    rng = np.random.default_rng(1000 + seed + D)  # deterministic per-D, distinct from data seeds
    return scale * rng.standard_normal(D) / np.sqrt(D)


# ==============================
# Plain (non-variational) Gaussian beta-loss, for the Laplace MAP fit
# ==============================

def gaussian_beta_loss_pointwise(resid: jnp.ndarray, sigma2: jnp.ndarray, beta: float) -> jnp.ndarray:
    """Closed-form pointwise beta-divergence loss for a Gaussian likelihood
    N(mu, sigma2), beta > 1 (the Gaussian case has an exact closed form for the
    normalizing term -- no truncation/asymptotic approximation needed, unlike
    the Poisson case).
    """
    log_pdf = -0.5 * jnp.log(2.0 * jnp.pi * sigma2) - resid**2 / (2.0 * sigma2)
    term1 = -(1.0 / (beta - 1.0)) * jnp.exp((beta - 1.0) * log_pdf)
    term2 = (1.0 / beta) * (2.0 * jnp.pi * sigma2) ** (-(beta - 1.0) / 2.0) * beta ** (-0.5)
    return term1 + term2


def _neg_beta_log_posterior(params: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray, beta: float,
                              a0: float, b0: float, V0_inv_diag: jnp.ndarray, p: int) -> jnp.ndarray:
    """params = concat([delta (p,), ell]), ell = log(sigma2). V0 assumed diagonal
    (V0 = sigma0^2 * I here), so V0_inv is passed as its diagonal for speed."""
    delta = params[:p]
    ell = params[p]
    sigma2 = jnp.exp(ell)
    resid = y - X @ delta

    if beta == 1.0:
        nll_data = jnp.sum(0.5 * jnp.log(2.0 * jnp.pi * sigma2) + resid**2 / (2.0 * sigma2))
    else:
        nll_data = jnp.sum(gaussian_beta_loss_pointwise(resid, sigma2, beta))

    neg_log_prior_sigma2 = (a0 + 1.0) * ell + b0 * jnp.exp(-ell)
    neg_log_prior_delta = 0.5 * p * ell + 0.5 * jnp.exp(-ell) * jnp.sum(V0_inv_diag * delta**2)

    return nll_data + neg_log_prior_sigma2 + neg_log_prior_delta


_nlp_grad = jax.jit(jax.grad(_neg_beta_log_posterior), static_argnums=(3, 7))
_nlp_val_grad = jax.jit(jax.value_and_grad(_neg_beta_log_posterior), static_argnums=(3, 7))
_nlp_hessian = jax.jit(jax.hessian(_neg_beta_log_posterior), static_argnums=(3, 7))


def laplace_beta_posterior(
    X: Array, y: Array, beta: float, a0: float, b0: float, sigma0: float,
    init: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """MAP + Laplace (Gaussian) approximation of the beta-generalized posterior
    over (delta, log_sigma2), via L-BFGS-B + exact JAX Hessian at the MAP.

    Returns (map_point, cov) where cov = inverse Hessian of the negative
    log-posterior at the MAP -- ready to feed directly into
    nig_beta_overlap_utils.minus2logBC_mvn for comparing two such
    approximations.
    """
    X = jnp.asarray(X, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    n, p = X.shape
    V0_inv_diag = jnp.full((p,), 1.0 / sigma0**2, dtype=jnp.float64)

    if init is None:
        # OLS-ish init: ridge solution + log residual variance
        XtX = np.asarray(X.T @ X) + 1e-6 * np.eye(p)
        delta0 = np.linalg.solve(XtX, np.asarray(X.T @ y))
        resid0 = np.asarray(y) - np.asarray(X) @ delta0
        ell0 = np.log(max(np.var(resid0), 1e-6))
        init = np.concatenate([delta0, [ell0]])

    def _fun_grad(params_np):
        params = jnp.asarray(params_np, dtype=jnp.float64)
        val, grad = _nlp_val_grad(params, X, y, float(beta), a0, b0, V0_inv_diag, p)
        return float(val), np.asarray(grad, dtype=np.float64)

    result = minimize(_fun_grad, init, jac=True, method="L-BFGS-B",
                       options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-8})
    map_point = result.x

    H = np.asarray(_nlp_hessian(jnp.asarray(map_point, dtype=jnp.float64), X, y, float(beta),
                                  a0, b0, V0_inv_diag, p), dtype=np.float64)
    H = 0.5 * (H + H.T)  # symmetrize (numerical hygiene)
    jitter = 1e-8
    for _ in range(8):
        try:
            cov = np.linalg.inv(H + jitter * np.eye(p + 1))
            np.linalg.cholesky(cov)  # verify PD
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:
        raise np.linalg.LinAlgError("Laplace Hessian not invertible/PD even with jitter")

    return map_point, cov


# ==============================
# Well-specified threshold (Section 2) and discrepancy curve (Section 3)
# ==============================

def well_specified_threshold(D: int, N: int, a0: float, b0: float, sigma0: float,
                               beta_true: Array, M: int, seed: int) -> Tuple[float, float, Array]:
    """d*_{1/2}: mean d_1/2 between M pairs of EXACT NIG posteriors, each pair
    fit on independent Gaussian-noise (variance-matched to t(5)) datasets."""
    ds = np.empty(M)
    for m in range(M):
        Xa, ya = generate_regression_data(D, N, seed=2 * (seed * M + m), beta_true=beta_true, noise="gaussian_matched")
        Xb, yb = generate_regression_data(D, N, seed=2 * (seed * M + m) + 1, beta_true=beta_true, noise="gaussian_matched")
        pa = nbu.fit_standard_nig_posterior(Xa, ya, a0=a0, b0=b0, sigma0=sigma0)
        pb = nbu.fit_standard_nig_posterior(Xb, yb, a0=a0, b0=b0, sigma0=sigma0)
        d12, _ = nbu.hellinger_nig_closed_form(pa, pb)
        ds[m] = d12
    return float(ds.mean()), float(ds.std(ddof=1) / np.sqrt(M)), ds


def observed_discrepancy_laplace(D: int, N: int, beta: float, a0: float, b0: float, sigma0: float,
                                    beta_true: Array, M: int, seed: int) -> Tuple[float, float, Array]:
    """d~_{1/2}(beta): mean d_1/2 between M pairs of Laplace-approximated
    beta-posteriors, each pair fit on independent t(5)-noise (misspecified)
    datasets."""
    ds = np.empty(M)
    for m in range(M):
        Xa, ya = generate_regression_data(D, N, seed=2 * (seed * M + m), beta_true=beta_true, noise="t5")
        Xb, yb = generate_regression_data(D, N, seed=2 * (seed * M + m) + 1, beta_true=beta_true, noise="t5")
        map_a, cov_a = laplace_beta_posterior(Xa, ya, beta, a0, b0, sigma0)
        map_b, cov_b = laplace_beta_posterior(Xb, yb, beta, a0, b0, sigma0)
        ds[m] = nbu.minus2logBC_mvn(map_a, cov_a, map_b, cov_b)
    return float(ds.mean()), float(ds.std(ddof=1) / np.sqrt(M)), ds


# ==============================
# GVI (diagonal NIG variational family) approximation of the beta-posterior
# ==============================

def gvi_beta_posterior_diag(
    X: Array, y: Array, beta: float, a0: float, b0: float, sigma0: float,
    n_epochs: int = 1000, lr: float = 0.02, init: Optional[Dict] = None,
) -> Dict[str, object]:
    """Diagonal-covariance NIG variational fit: minimize E_q[beta-loss] +
    KL(q||prior) via Adam, reusing nig_beta_overlap_utils' expected_beta_loss /
    kl_nig (both accept any square V, diagonal included). Returns a genuine
    NIGParams dict (mu, V=diag(v), d, beta) -- directly comparable via the
    exact closed-form NIG Bhattacharyya coefficient, same as the well-specified
    threshold, unlike Laplace's generic (delta, log-sigma2) Gaussian output.
    """
    X_t = torch.as_tensor(X, dtype=torch.float64)
    y_t = torch.as_tensor(y, dtype=torch.float64)
    n, p = X_t.shape

    mu_0 = torch.zeros(p, dtype=torch.float64)
    V_0 = (sigma0**2) * torch.eye(p, dtype=torch.float64)
    d_0 = torch.tensor(a0, dtype=torch.float64)
    beta_0 = torch.tensor(b0, dtype=torch.float64)

    if init is not None:
        mu_init = torch.as_tensor(init["mu"], dtype=torch.float64).clone()
        v_init = torch.as_tensor(np.diag(np.asarray(init["V"])), dtype=torch.float64).clamp(min=1e-8)
        d_init = torch.as_tensor(float(init["d"]), dtype=torch.float64)
        beta_init = torch.as_tensor(float(init["beta"]), dtype=torch.float64)
    else:
        mu_init = mu_0.clone()
        v_init = torch.diag(V_0).clone()
        d_init = d_0.clone()
        beta_init = beta_0.clone()

    q_mu = mu_init.clone().requires_grad_(True)
    q_log_v = torch.log(v_init).clone().requires_grad_(True)
    q_log_d = torch.log(d_init).clone().requires_grad_(True)
    q_log_beta = torch.log(beta_init).clone().requires_grad_(True)

    optimizer = torch.optim.Adam([q_mu, q_log_v, q_log_d, q_log_beta], lr=lr)

    for _ in range(n_epochs):
        optimizer.zero_grad()
        q_V = torch.diag(torch.exp(q_log_v))
        q_d = torch.exp(q_log_d)
        q_beta = torch.exp(q_log_beta)
        data_term = nbu.expected_beta_loss(X_t, y_t, q_mu, q_V, q_d, q_beta, beta)
        kl_term = nbu.kl_nig(q_mu, q_V, q_d, q_beta, mu_0, V_0, d_0, beta_0)
        loss = data_term + kl_term
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return {
            "mu": q_mu.detach().numpy(),
            "V": torch.diag(torch.exp(q_log_v)).detach().numpy(),
            "d": float(torch.exp(q_log_d).item()),
            "beta": float(torch.exp(q_log_beta).item()),
        }


def observed_discrepancy_gvi(D: int, N: int, beta: float, a0: float, b0: float, sigma0: float,
                               beta_true: Array, M: int, seed: int,
                               n_epochs: int = 1000, lr: float = 0.02) -> Tuple[float, float, Array]:
    """d~_{1/2}(beta) via diagonal GVI instead of Laplace, using the exact
    closed-form NIG Bhattacharyya coefficient (GVI output is genuine NIG form)."""
    ds = np.empty(M)
    for m in range(M):
        Xa, ya = generate_regression_data(D, N, seed=2 * (seed * M + m), beta_true=beta_true, noise="t5")
        Xb, yb = generate_regression_data(D, N, seed=2 * (seed * M + m) + 1, beta_true=beta_true, noise="t5")
        params_a = gvi_beta_posterior_diag(Xa, ya, beta, a0, b0, sigma0, n_epochs=n_epochs, lr=lr)
        params_b = gvi_beta_posterior_diag(Xb, yb, beta, a0, b0, sigma0, n_epochs=n_epochs, lr=lr)
        d12, _ = nbu.hellinger_nig_closed_form(params_a, params_b)
        ds[m] = d12
    return float(ds.mean()), float(ds.std(ddof=1) / np.sqrt(M)), ds


# ================================================================
# SIMPLIFIED experiment: known/fixed sigma2, pure variance misspecification.
#
# Model: y = X @ delta + N(0, sigma2_assumed), delta ~ N(0, diag(V0)). sigma2 is
# FIXED/known (not estimated) -- this removes the (delta, log-sigma2) coupling
# that caused Laplace's variance-collapse issue, and removes the t(5) heavy
# tails that drove GVI's KL-vs-data-term imbalance. Misspecification is now
# purely: the model assumes sigma2_assumed, but the true DGP uses sigma2_true.
#
# Well-specified threshold and standard-Bayes (beta=1) discrepancy are EXACT
# closed-form Gaussian conjugate updates -- no approximation needed at all.
# Only the beta!=1 generalized posterior over delta needs Laplace or GVI.
# ================================================================

def generate_variance_misspec_data(D: int, N: int, seed: int, delta_true: Array, sigma2: float) -> Tuple[Array, Array]:
    """iid standard-normal design X (N x D); y = X @ delta_true + N(0, sigma2)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, D))
    y = X @ delta_true + rng.normal(scale=np.sqrt(sigma2), size=N)
    return X, y


def exact_gaussian_posterior_knownvar(X: Array, y: Array, sigma2_assumed: float, v0: Array) -> Tuple[Array, Array]:
    """Exact closed-form posterior over delta: prior N(0, diag(v0)), likelihood
    N(y; X@delta, sigma2_assumed) with sigma2_assumed treated as KNOWN (not
    estimated). Returns (mu_n, V_n)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    V0_inv = np.diag(1.0 / v0)
    Vn_inv = V0_inv + (X.T @ X) / sigma2_assumed
    Vn = np.linalg.inv(Vn_inv)
    mu_n = Vn @ (X.T @ y) / sigma2_assumed
    return mu_n, Vn


def well_specified_threshold_knownvar(D: int, N: int, sigma2_assumed: float, v0: Array,
                                        delta_true: Array, M: int, seed: int) -> Tuple[float, float, Array]:
    """d*_{1/2}: mean d_1/2 between M pairs of EXACT Gaussian posteriors, each
    fit on data whose TRUE variance matches the assumed sigma2_assumed (no
    misspecification) -- fully closed-form, no approximation."""
    ds = np.empty(M)
    for m in range(M):
        Xa, ya = generate_variance_misspec_data(D, N, seed=2 * (seed * M + m), delta_true=delta_true, sigma2=sigma2_assumed)
        Xb, yb = generate_variance_misspec_data(D, N, seed=2 * (seed * M + m) + 1, delta_true=delta_true, sigma2=sigma2_assumed)
        mu_a, V_a = exact_gaussian_posterior_knownvar(Xa, ya, sigma2_assumed, v0)
        mu_b, V_b = exact_gaussian_posterior_knownvar(Xb, yb, sigma2_assumed, v0)
        ds[m] = nbu.minus2logBC_mvn(mu_a, V_a, mu_b, V_b)
    return float(ds.mean()), float(ds.std(ddof=1) / np.sqrt(M)), ds


def observed_discrepancy_std_bayes_knownvar(D: int, N: int, sigma2_assumed: float, sigma2_true: float,
                                              v0: Array, delta_true: Array, M: int, seed: int) -> Tuple[float, float, Array]:
    """d_obs under standard Bayes (beta=1): same exact closed-form posterior
    machinery, but now fit on data generated with the MISMATCHED sigma2_true --
    still exact/closed-form (beta=1 never needed the beta-loss machinery), just
    honestly miscalibrated relative to the true generating uncertainty."""
    ds = np.empty(M)
    for m in range(M):
        Xa, ya = generate_variance_misspec_data(D, N, seed=2 * (seed * M + m), delta_true=delta_true, sigma2=sigma2_true)
        Xb, yb = generate_variance_misspec_data(D, N, seed=2 * (seed * M + m) + 1, delta_true=delta_true, sigma2=sigma2_true)
        mu_a, V_a = exact_gaussian_posterior_knownvar(Xa, ya, sigma2_assumed, v0)
        mu_b, V_b = exact_gaussian_posterior_knownvar(Xb, yb, sigma2_assumed, v0)
        ds[m] = nbu.minus2logBC_mvn(mu_a, V_a, mu_b, V_b)
    return float(ds.mean()), float(ds.std(ddof=1) / np.sqrt(M)), ds


# ---- Laplace approximation (delta only, sigma2 fixed) ----

def _neg_beta_log_posterior_knownvar(delta: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray, beta: float,
                                       sigma2: float, v0_inv: jnp.ndarray) -> jnp.ndarray:
    resid = y - X @ delta
    if beta == 1.0:
        nll_data = jnp.sum(0.5 * jnp.log(2.0 * jnp.pi * sigma2) + resid**2 / (2.0 * sigma2))
    else:
        nll_data = jnp.sum(gaussian_beta_loss_pointwise(resid, sigma2, beta))
    neg_log_prior = 0.5 * jnp.sum(v0_inv * delta**2)
    return nll_data + neg_log_prior


_nlp_kv_val_grad = jax.jit(jax.value_and_grad(_neg_beta_log_posterior_knownvar), static_argnums=(3,))
_nlp_kv_hessian = jax.jit(jax.hessian(_neg_beta_log_posterior_knownvar), static_argnums=(3,))


def laplace_beta_posterior_knownvar(X: Array, y: Array, beta: float, sigma2: float, v0: Array) -> Tuple[Array, Array]:
    """MAP + Laplace approximation over delta ONLY (sigma2 fixed/known) --
    much lower-dimensional and better-behaved than the unknown-variance case."""
    X_j = jnp.asarray(X, dtype=jnp.float64)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    n, p = X_j.shape
    v0_inv = jnp.asarray(1.0 / v0, dtype=jnp.float64)

    XtX = np.asarray(X).T @ np.asarray(X) + 1e-6 * np.eye(p)
    delta0 = np.linalg.solve(XtX, np.asarray(X).T @ np.asarray(y))

    def _fun_grad(delta_np):
        delta = jnp.asarray(delta_np, dtype=jnp.float64)
        val, grad = _nlp_kv_val_grad(delta, X_j, y_j, float(beta), sigma2, v0_inv)
        return float(val), np.asarray(grad, dtype=np.float64)

    result = minimize(_fun_grad, delta0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-8})
    map_point = result.x

    H = np.asarray(_nlp_kv_hessian(jnp.asarray(map_point, dtype=jnp.float64), X_j, y_j, float(beta),
                                     sigma2, v0_inv), dtype=np.float64)
    H = 0.5 * (H + H.T)
    jitter = 1e-8
    for _ in range(8):
        try:
            cov = np.linalg.inv(H + jitter * np.eye(p))
            np.linalg.cholesky(cov)
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:
        raise np.linalg.LinAlgError("Laplace Hessian not invertible/PD even with jitter")

    return map_point, cov


def observed_discrepancy_laplace_knownvar(D: int, N: int, beta: float, sigma2_assumed: float, sigma2_true: float,
                                            v0: Array, delta_true: Array, M: int, seed: int) -> Tuple[float, float, Array]:
    ds = np.empty(M)
    for m in range(M):
        Xa, ya = generate_variance_misspec_data(D, N, seed=2 * (seed * M + m), delta_true=delta_true, sigma2=sigma2_true)
        Xb, yb = generate_variance_misspec_data(D, N, seed=2 * (seed * M + m) + 1, delta_true=delta_true, sigma2=sigma2_true)
        map_a, cov_a = laplace_beta_posterior_knownvar(Xa, ya, beta, sigma2_assumed, v0)
        map_b, cov_b = laplace_beta_posterior_knownvar(Xb, yb, beta, sigma2_assumed, v0)
        ds[m] = nbu.minus2logBC_mvn(map_a, cov_a, map_b, cov_b)
    return float(ds.mean()), float(ds.std(ddof=1) / np.sqrt(M)), ds


# ---- GVI (diagonal Gaussian over delta only, sigma2 fixed) ----
# Closed-form E_q[beta-loss]: mu_i = x_i^T delta is Gaussian under q (since q is
# Gaussian), so E_q[f(y_i;mu_i,sigma2)^(beta-1)] reduces to a standard Gaussian
# integral identity E[exp(-a(c-Z)^2)] = (1+2as^2)^{-1/2} exp(-a(c-m)^2/(1+2as^2))
# for Z~N(m,s^2) -- no Monte Carlo needed, unlike the unknown-variance NIG case.

def gvi_beta_posterior_knownvar_diag(
    X: Array, y: Array, beta: float, sigma2: float, v0: Array,
    n_epochs: int = 500, lr: float = 0.05,
) -> Tuple[Array, Array]:
    X_t = torch.as_tensor(X, dtype=torch.float64)
    y_t = torch.as_tensor(y, dtype=torch.float64)
    n, p = X_t.shape
    v0_t = torch.as_tensor(v0, dtype=torch.float64)

    XtX = X_t.T @ X_t + 1e-6 * torch.eye(p, dtype=torch.float64)
    delta0 = torch.linalg.solve(XtX, X_t.T @ y_t)

    q_mu = delta0.clone().requires_grad_(True)
    q_log_v = torch.log(v0_t).clone().requires_grad_(True)

    optimizer = torch.optim.Adam([q_mu, q_log_v], lr=lr)

    for _ in range(n_epochs):
        optimizer.zero_grad()
        q_v = torch.exp(q_log_v)
        m_i = X_t @ q_mu
        s2_i = (X_t**2) @ q_v  # Var[mu_i] under q, per observation

        if beta == 1.0:
            data_term = torch.sum(0.5 * torch.log(torch.as_tensor(2.0 * np.pi * sigma2, dtype=torch.float64)) + ((y_t - m_i)**2 + s2_i) / (2.0 * sigma2))
        else:
            a = (beta - 1.0) / (2.0 * sigma2)
            denom = 1.0 + 2.0 * a * s2_i
            log_pref = -(beta - 1.0) / 2.0 * torch.log(2.0 * np.pi * torch.as_tensor(sigma2, dtype=torch.float64))
            e_term1_i = torch.exp(log_pref) / torch.sqrt(denom) * torch.exp(-a * (y_t - m_i)**2 / denom)
            term1 = -(1.0 / (beta - 1.0)) * e_term1_i
            term2 = (1.0 / beta) * (2.0 * np.pi * sigma2) ** (-(beta - 1.0) / 2.0) * beta ** (-0.5)
            data_term = torch.sum(term1 + term2)

        # KL(N(q_mu,diag(q_v)) || N(0,diag(v0))) -- diagonal, so a plain sum
        kl_term = 0.5 * torch.sum(q_v / v0_t + q_mu**2 / v0_t - 1.0 + torch.log(v0_t / q_v))

        loss = data_term + kl_term
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        mu_final = q_mu.detach().numpy()
        V_final = np.diag(torch.exp(q_log_v).detach().numpy())
    return mu_final, V_final


def observed_discrepancy_gvi_knownvar(D: int, N: int, beta: float, sigma2_assumed: float, sigma2_true: float,
                                        v0: Array, delta_true: Array, M: int, seed: int,
                                        n_epochs: int = 500, lr: float = 0.05) -> Tuple[float, float, Array]:
    ds = np.empty(M)
    for m in range(M):
        Xa, ya = generate_variance_misspec_data(D, N, seed=2 * (seed * M + m), delta_true=delta_true, sigma2=sigma2_true)
        Xb, yb = generate_variance_misspec_data(D, N, seed=2 * (seed * M + m) + 1, delta_true=delta_true, sigma2=sigma2_true)
        mu_a, V_a = gvi_beta_posterior_knownvar_diag(Xa, ya, beta, sigma2_assumed, v0, n_epochs=n_epochs, lr=lr)
        mu_b, V_b = gvi_beta_posterior_knownvar_diag(Xb, yb, beta, sigma2_assumed, v0, n_epochs=n_epochs, lr=lr)
        ds[m] = nbu.minus2logBC_mvn(mu_a, V_a, mu_b, V_b)
    return float(ds.mean()), float(ds.std(ddof=1) / np.sqrt(M)), ds


# ==============================
# beta* root-find: bracket a sign change in d~_1/2(beta) - d*_1/2, refine via brentq
# ==============================

def find_beta_star(discrepancy_fn, d_star: float,
                     beta_candidates: Tuple[float, ...] = (1.02, 1.1, 1.3, 1.6, 2.0, 2.5, 3.0),
                     xtol: float = 1e-3) -> Tuple[Optional[float], list]:
    """discrepancy_fn(beta) -> (mean_d, se_d, ds). Evaluates on beta_candidates
    to find a sign change in (d~_1/2(beta) - d*_1/2), then refines with brentq.
    Returns (beta_star_or_None, [(beta, d_mean, gap), ...]) for diagnostics."""
    from scipy.optimize import brentq

    curve = []
    gaps = []
    for b in beta_candidates:
        d_mean, se, _ = discrepancy_fn(b)
        gap = d_mean - d_star
        curve.append((b, d_mean, gap))
        gaps.append(gap)

    for i in range(len(beta_candidates) - 1):
        if gaps[i] == 0.0:
            return beta_candidates[i], curve
        if gaps[i] * gaps[i + 1] < 0:
            def gap_fn(b):
                d_mean, se, _ = discrepancy_fn(b)
                return d_mean - d_star
            beta_star = brentq(gap_fn, beta_candidates[i], beta_candidates[i + 1], xtol=xtol)
            return float(beta_star), curve

    return None, curve  # no sign change found in the candidate range


# ==============================
# Ground-truth check: NUTS on the actual beta-posterior, compared to Laplace
# ==============================

def _numpyro_model_knownvar(X, y, beta, sigma2, v0):
    delta = numpyro.sample("delta", dist.Normal(0.0, jnp.sqrt(v0)).to_event(1))
    mu = X @ delta
    if beta == 1.0:
        numpyro.sample("obs", dist.Normal(mu, jnp.sqrt(sigma2)), obs=y)
    else:
        resid = y - mu
        neg_loss = -jnp.sum(gaussian_beta_loss_pointwise(resid, sigma2, beta))
        numpyro.factor("beta_loglik", neg_loss)


def run_nuts_knownvar(X: Array, y: Array, beta: float, sigma2: float, v0: Array,
                        num_warmup: int = 1000, num_samples: int = 2000, seed: int = 0,
                        target_accept_prob: float = 0.9) -> Tuple[Array, float]:
    """Ground-truth samples from the actual beta-posterior over delta (no
    Laplace/GVI approximation), via NUTS -- same toolchain as the Poisson
    chapters. Returns (samples (S,p), divergence_rate)."""
    X_j = jnp.asarray(X, dtype=jnp.float64)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    v0_j = jnp.asarray(v0, dtype=jnp.float64)
    kernel = NUTS(_numpyro_model_knownvar, target_accept_prob=target_accept_prob)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), X_j, y_j, float(beta), float(sigma2), v0_j)
    div_rate = float(np.asarray(mcmc.get_extra_fields()["diverging"]).mean())
    return np.asarray(mcmc.get_samples()["delta"]), div_rate


def compare_laplace_to_nuts(X: Array, y: Array, beta: float, sigma2: float, v0: Array,
                              num_warmup: int = 1000, num_samples: int = 2000, seed: int = 0) -> Dict:
    """Fits both Laplace and NUTS on the SAME dataset/beta, and reports how far
    apart they are: mean difference, a quantifiable d_1/2 between Laplace's
    Gaussian and the NUTS-sample moment-matched Gaussian, and the NUTS samples'
    own marginal skew/kurtosis (Gaussian => 0, 0) to directly check whether the
    true posterior is even approximately Gaussian-shaped."""
    from scipy.stats import skew, kurtosis

    map_point, cov_laplace = laplace_beta_posterior_knownvar(X, y, beta, sigma2, v0)
    nuts_samples, div_rate = run_nuts_knownvar(X, y, beta, sigma2, v0, num_warmup=num_warmup,
                                                 num_samples=num_samples, seed=seed)
    nuts_mean = nuts_samples.mean(axis=0)
    nuts_cov = np.cov(nuts_samples, rowvar=False)
    d_disc = nbu.minus2logBC_mvn(map_point, cov_laplace, nuts_mean, nuts_cov)
    marg_skew = skew(nuts_samples, axis=0)
    marg_exkurt = kurtosis(nuts_samples, axis=0)

    return {
        "mean_abs_diff": float(np.mean(np.abs(map_point - nuts_mean))),
        "mean_rel_diff": float(np.mean(np.abs(map_point - nuts_mean) / (np.abs(nuts_mean) + 1e-8))),
        "cov_diag_ratio_laplace_over_nuts": (np.diag(cov_laplace) / np.diag(nuts_cov)).tolist(),
        "d_half_laplace_vs_nuts": float(d_disc),
        "nuts_div_rate": div_rate,
        "marginal_skew_min_max": (float(marg_skew.min()), float(marg_skew.max())),
        "marginal_exkurt_min_max": (float(marg_exkurt.min()), float(marg_exkurt.max())),
    }


# ==============================
# D=1 exact posterior via numerical quadrature (trapezoid rule) -- no Laplace
# or MCMC approximation at all, only grid discretization error, which can be
# made arbitrarily small. Only tractable for very small D (curse of
# dimensionality for grid quadrature).
# ==============================

def exact_posterior_1d_quadrature(X: Array, y: Array, beta: float, sigma2: float, v0: float,
                                    grid_half_width: float = 6.0, n_grid: int = 20001) -> Tuple[float, float]:
    """X: (N,1), y: (N,). Returns (exact_mean, exact_var) of the D=1
    beta-posterior over delta, via trapezoid-rule quadrature on a fine grid."""
    X = np.asarray(X, dtype=float).reshape(-1)  # (N,)
    y = np.asarray(y, dtype=float)
    delta_grid = np.linspace(-grid_half_width, grid_half_width, n_grid)

    mu = X[:, None] * delta_grid[None, :]  # (N, G)
    resid = y[:, None] - mu

    if beta == 1.0:
        pointwise = 0.5 * np.log(2.0 * np.pi * sigma2) + resid**2 / (2.0 * sigma2)
    else:
        mu_j = jnp.asarray(mu)
        resid_j = jnp.asarray(resid)
        pointwise = np.asarray(gaussian_beta_loss_pointwise(resid_j, sigma2, beta))

    nll_data = pointwise.sum(axis=0)  # (G,)
    log_prior = -0.5 * delta_grid**2 / v0  # up to an additive constant, irrelevant after normalizing
    log_unnorm = -nll_data + log_prior
    log_unnorm -= log_unnorm.max()
    density_unnorm = np.exp(log_unnorm)

    Z = np.trapz(density_unnorm, delta_grid)
    density = density_unnorm / Z
    mean = np.trapz(delta_grid * density, delta_grid)
    var = np.trapz((delta_grid - mean) ** 2 * density, delta_grid)
    return float(mean), float(var)


# ==============================
# GVI with continuation (warm-started from beta near 1, stepping up gradually)
# -- mirrors nig_beta_overlap_utils.gvi_beta_posterior_continuation but for the
# known-variance diagonal model, to distinguish genuine KL-scaling collapse
# from mere under-optimization/cold-start failure.
# ==============================

def gvi_beta_posterior_knownvar_continuation(
    X: Array, y: Array, beta: float, sigma2: float, v0: Array,
    n_epochs_per_step: int = 300, lr: float = 0.03,
    beta_anchor: float = 1.0 + 1e-6, max_step: float = 0.05,
) -> Tuple[Array, Array]:
    """Fits a monotone beta grid from beta_anchor (~1) up to beta, each step
    warm-started from the previous step's converged solution, rather than
    cold-starting directly at the target beta."""
    beta = float(beta)
    if beta <= beta_anchor:
        return gvi_beta_posterior_knownvar_diag(X, y, beta, sigma2, v0, n_epochs=n_epochs_per_step, lr=lr)

    n_steps = max(2, int(np.ceil((beta - beta_anchor) / max_step)) + 1)
    beta_grid = np.linspace(beta_anchor, beta, n_steps)

    X_t = torch.as_tensor(X, dtype=torch.float64)
    y_t = torch.as_tensor(y, dtype=torch.float64)
    n, p = X_t.shape
    v0_t = torch.as_tensor(v0, dtype=torch.float64)

    XtX = X_t.T @ X_t + 1e-6 * torch.eye(p, dtype=torch.float64)
    q_mu = torch.linalg.solve(XtX, X_t.T @ y_t).clone().requires_grad_(True)
    q_log_v = torch.log(v0_t).clone().requires_grad_(True)

    for beta_step in beta_grid:
        beta_step = float(beta_step)
        optimizer = torch.optim.Adam([q_mu, q_log_v], lr=lr)
        for _ in range(n_epochs_per_step):
            optimizer.zero_grad()
            q_v = torch.exp(q_log_v)
            m_i = X_t @ q_mu
            s2_i = (X_t**2) @ q_v

            if beta_step == 1.0:
                data_term = torch.sum(0.5 * torch.log(torch.as_tensor(2.0 * np.pi * sigma2, dtype=torch.float64)) + ((y_t - m_i)**2 + s2_i) / (2.0 * sigma2))
            else:
                a = (beta_step - 1.0) / (2.0 * sigma2)
                denom = 1.0 + 2.0 * a * s2_i
                log_pref = -(beta_step - 1.0) / 2.0 * torch.log(2.0 * np.pi * torch.as_tensor(sigma2, dtype=torch.float64))
                e_term1_i = torch.exp(log_pref) / torch.sqrt(denom) * torch.exp(-a * (y_t - m_i)**2 / denom)
                term1 = -(1.0 / (beta_step - 1.0)) * e_term1_i
                term2 = (1.0 / beta_step) * (2.0 * np.pi * sigma2) ** (-(beta_step - 1.0) / 2.0) * beta_step ** (-0.5)
                data_term = torch.sum(term1 + term2)

            kl_term = 0.5 * torch.sum(q_v / v0_t + q_mu**2 / v0_t - 1.0 + torch.log(v0_t / q_v))
            loss = data_term + kl_term
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        mu_final = q_mu.detach().numpy()
        V_final = np.diag(torch.exp(q_log_v).detach().numpy())
    return mu_final, V_final


def observed_discrepancy_gvi_knownvar_continuation(
    D: int, N: int, beta: float, sigma2_assumed: float, sigma2_true: float,
    v0: Array, delta_true: Array, M: int, seed: int,
    n_epochs_per_step: int = 300, lr: float = 0.03, max_step: float = 0.05,
) -> Tuple[float, float, Array]:
    ds = np.empty(M)
    for m in range(M):
        Xa, ya = generate_variance_misspec_data(D, N, seed=2 * (seed * M + m), delta_true=delta_true, sigma2=sigma2_true)
        Xb, yb = generate_variance_misspec_data(D, N, seed=2 * (seed * M + m) + 1, delta_true=delta_true, sigma2=sigma2_true)
        mu_a, V_a = gvi_beta_posterior_knownvar_continuation(Xa, ya, beta, sigma2_assumed, v0,
                                                                n_epochs_per_step=n_epochs_per_step, lr=lr, max_step=max_step)
        mu_b, V_b = gvi_beta_posterior_knownvar_continuation(Xb, yb, beta, sigma2_assumed, v0,
                                                                n_epochs_per_step=n_epochs_per_step, lr=lr, max_step=max_step)
        ds[m] = nbu.minus2logBC_mvn(mu_a, V_a, mu_b, V_b)
    return float(ds.mean()), float(ds.std(ddof=1) / np.sqrt(M)), ds


# ==============================
# NUTS-based discrepancy (ground truth for the beta* search itself, not just
# single-fit comparisons) -- for low-D cells only, given the cost.
# ==============================

def observed_discrepancy_nuts_knownvar(D: int, N: int, beta: float, sigma2_assumed: float, sigma2_true: float,
                                         v0: Array, delta_true: Array, M: int, seed: int,
                                         num_warmup: int = 1000, num_samples: int = 2000,
                                         target_accept_prob: float = 0.9) -> Tuple[float, float, Array]:
    """d~_{1/2}(beta) via NUTS instead of Laplace/GVI -- the actual ground-truth
    discrepancy, at real cost. Uses each fit's full sample mean/covariance,
    compared via the closed-form Gaussian BC on those Gaussian moment-matches
    (same comparison mechanism as Laplace/GVI, so directly comparable).

    Builds ONE NUTS kernel/MCMC object and reuses it (via repeated .run()
    calls) across all 2*M replicate fits, instead of constructing a fresh
    MCMC object per fit (the original approach) -- D, N, beta, sigma2, v0 are
    constant across the loop, only the data and seed change, so a single
    object lets JAX JIT-compile once and reuse that compilation. Constructing
    fresh MCMC objects per call was observed to make JAX/XLA's compilation
    cache grow without bound across the loop (~200 fresh compiles per cell at
    M=100), causing steady, eventually fatal RSS growth that got noticeably
    worse as D grew (D=50 died before finishing; D=10/20 were small enough to
    finish before hitting a wall)."""
    import gc
    v0_j = jnp.asarray(v0, dtype=jnp.float64)
    kernel = NUTS(_numpyro_model_knownvar, target_accept_prob=target_accept_prob)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=1, progress_bar=False)

    ds = np.empty(M)
    for m in range(M):
        Xa, ya = generate_variance_misspec_data(D, N, seed=2 * (seed * M + m), delta_true=delta_true, sigma2=sigma2_true)
        Xb, yb = generate_variance_misspec_data(D, N, seed=2 * (seed * M + m) + 1, delta_true=delta_true, sigma2=sigma2_true)

        mcmc.run(jax.random.PRNGKey(seed), jnp.asarray(Xa, dtype=jnp.float64), jnp.asarray(ya, dtype=jnp.float64), float(beta), float(sigma2_assumed), v0_j)
        samples_a_jax = mcmc.get_samples()["delta"]
        samples_a = np.array(samples_a_jax)
        del samples_a_jax

        mcmc.run(jax.random.PRNGKey(seed + 1), jnp.asarray(Xb, dtype=jnp.float64), jnp.asarray(yb, dtype=jnp.float64), float(beta), float(sigma2_assumed), v0_j)
        samples_b_jax = mcmc.get_samples()["delta"]
        samples_b = np.array(samples_b_jax)
        del samples_b_jax

        mean_a, cov_a = samples_a.mean(axis=0), np.cov(samples_a, rowvar=False)
        mean_b, cov_b = samples_b.mean(axis=0), np.cov(samples_b, rowvar=False)
        ds[m] = nbu.minus2logBC_mvn(mean_a, cov_a, mean_b, cov_b)

        # JAX's allocator doesn't release device buffers just because the
        # Python object went out of scope -- without periodic clearing, RSS
        # grows monotonically across the M-replicate loop (observed directly:
        # ~150-300MB per 20s at D=50, accelerating, eventually OOM-killing
        # the worker before a cell finishes). Every 20 replicates, drop
        # JAX's compilation/buffer caches and force a GC pass.
        if (m + 1) % 20 == 0:
            jax.clear_caches()
            gc.collect()
    return float(ds.mean()), float(ds.std(ddof=1) / np.sqrt(M)), ds


# ==============================
# Full-covariance GVI (known-variance model): same objective as the diagonal
# version, but q gets a full Cholesky-parameterized covariance -- exact at
# beta=1 (a full-covariance Gaussian family can match the true Gaussian
# posterior exactly there), isolating the KL/data-term prior-collapse issue
# from the mean-field/diagonal correlational limitation.
# ==============================

def gvi_beta_posterior_knownvar_fullcov(
    X: Array, y: Array, beta: float, sigma2: float, v0: Array,
    n_epochs: int = 1000, lr: float = 0.02,
) -> Tuple[Array, Array]:
    X_t = torch.as_tensor(X, dtype=torch.float64)
    y_t = torch.as_tensor(y, dtype=torch.float64)
    n, p = X_t.shape
    v0_t = torch.as_tensor(v0, dtype=torch.float64)
    V0_t = torch.diag(v0_t)

    XtX = X_t.T @ X_t + 1e-6 * torch.eye(p, dtype=torch.float64)
    mu0 = torch.linalg.solve(XtX, X_t.T @ y_t)

    q_mu = mu0.clone().requires_grad_(True)
    L_init = torch.linalg.cholesky(V0_t)
    q_L_diag_log = torch.log(torch.diag(L_init)).clone().requires_grad_(True)
    q_L_offdiag = torch.zeros((p, p), dtype=torch.float64).requires_grad_(True)

    optimizer = torch.optim.Adam([q_mu, q_L_diag_log, q_L_offdiag], lr=lr)

    for _ in range(n_epochs):
        optimizer.zero_grad()
        L = torch.tril(q_L_offdiag, diagonal=-1) + torch.diag(torch.exp(q_L_diag_log))
        q_V = L @ L.T
        m_i = X_t @ q_mu
        s2_i = torch.sum((X_t @ q_V) * X_t, dim=1)  # Var[mu_i] under full-cov q

        if beta == 1.0:
            data_term = torch.sum(0.5 * torch.log(torch.as_tensor(2.0 * np.pi * sigma2, dtype=torch.float64)) + ((y_t - m_i)**2 + s2_i) / (2.0 * sigma2))
        else:
            a = (beta - 1.0) / (2.0 * sigma2)
            denom = 1.0 + 2.0 * a * s2_i
            log_pref = -(beta - 1.0) / 2.0 * torch.log(2.0 * np.pi * torch.as_tensor(sigma2, dtype=torch.float64))
            e_term1_i = torch.exp(log_pref) / torch.sqrt(denom) * torch.exp(-a * (y_t - m_i)**2 / denom)
            term1 = -(1.0 / (beta - 1.0)) * e_term1_i
            term2 = (1.0 / beta) * (2.0 * np.pi * sigma2) ** (-(beta - 1.0) / 2.0) * beta ** (-0.5)
            data_term = torch.sum(term1 + term2)

        # KL(N(q_mu,q_V) || N(0,V0)), V0 diagonal
        V0_inv_diag = 1.0 / v0_t
        logdet_q_V = 2.0 * torch.sum(q_L_diag_log)
        logdet_V0 = torch.sum(torch.log(v0_t))
        kl_term = 0.5 * (torch.sum(V0_inv_diag * torch.diag(q_V)) + torch.sum(q_mu**2 * V0_inv_diag)
                          - p + logdet_V0 - logdet_q_V)

        loss = data_term + kl_term
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        L = torch.tril(q_L_offdiag, diagonal=-1) + torch.diag(torch.exp(q_L_diag_log))
        mu_final = q_mu.detach().numpy()
        V_final = (L @ L.T).detach().numpy()
    return mu_final, V_final


def observed_discrepancy_gvi_knownvar_fullcov(D: int, N: int, beta: float, sigma2_assumed: float, sigma2_true: float,
                                                v0: Array, delta_true: Array, M: int, seed: int,
                                                n_epochs: int = 1000, lr: float = 0.02) -> Tuple[float, float, Array]:
    ds = np.empty(M)
    for m in range(M):
        Xa, ya = generate_variance_misspec_data(D, N, seed=2 * (seed * M + m), delta_true=delta_true, sigma2=sigma2_true)
        Xb, yb = generate_variance_misspec_data(D, N, seed=2 * (seed * M + m) + 1, delta_true=delta_true, sigma2=sigma2_true)
        mu_a, V_a = gvi_beta_posterior_knownvar_fullcov(Xa, ya, beta, sigma2_assumed, v0, n_epochs=n_epochs, lr=lr)
        mu_b, V_b = gvi_beta_posterior_knownvar_fullcov(Xb, yb, beta, sigma2_assumed, v0, n_epochs=n_epochs, lr=lr)
        ds[m] = nbu.minus2logBC_mvn(mu_a, V_a, mu_b, V_b)
    return float(ds.mean()), float(ds.std(ddof=1) / np.sqrt(M)), ds


# ==============================
# Low-rank Laplace: exact diagonal Hessian + rank-r correction from the top-r
# (by |eigenvalue|) eigenpairs of the off-diagonal residual H - diag(H) --
# captures the most significant correlations without needing a full
# unconstrained (D x D) covariance.
# ==============================

def low_rank_laplace_posterior_knownvar(X: Array, y: Array, beta: float, sigma2: float, v0: Array,
                                          rank: int = 15) -> Tuple[Array, Array]:
    map_point, _ = laplace_beta_posterior_knownvar(X, y, beta, sigma2, v0)
    X_j = jnp.asarray(X, dtype=jnp.float64)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    p = X.shape[1]
    v0_inv = jnp.asarray(1.0 / v0, dtype=jnp.float64)

    H = np.asarray(_nlp_kv_hessian(jnp.asarray(map_point, dtype=jnp.float64), X_j, y_j, float(beta),
                                     sigma2, v0_inv), dtype=np.float64)
    H = 0.5 * (H + H.T)

    H_diag = np.diag(np.diag(H))
    R = H - H_diag  # off-diagonal residual, zero diagonal
    eigvals, eigvecs = np.linalg.eigh(R)  # ascending
    order = np.argsort(-np.abs(eigvals))  # descending by magnitude
    r = min(rank, p)
    top_idx = order[:r]
    U_r = eigvecs[:, top_idx]
    lam_r = eigvals[top_idx]
    R_lowrank = U_r @ np.diag(lam_r) @ U_r.T

    H_lowrank = H_diag + R_lowrank
    H_lowrank = 0.5 * (H_lowrank + H_lowrank.T)
    jitter = 1e-8
    for _ in range(8):
        try:
            cov = np.linalg.inv(H_lowrank + jitter * np.eye(p))
            np.linalg.cholesky(cov)
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:
        raise np.linalg.LinAlgError("Low-rank Laplace Hessian not invertible/PD even with jitter")

    return map_point, cov


def observed_discrepancy_lowrank_laplace_knownvar(D: int, N: int, beta: float, sigma2_assumed: float, sigma2_true: float,
                                                     v0: Array, delta_true: Array, M: int, seed: int,
                                                     rank: int = 15) -> Tuple[float, float, Array]:
    ds = np.empty(M)
    for m in range(M):
        Xa, ya = generate_variance_misspec_data(D, N, seed=2 * (seed * M + m), delta_true=delta_true, sigma2=sigma2_true)
        Xb, yb = generate_variance_misspec_data(D, N, seed=2 * (seed * M + m) + 1, delta_true=delta_true, sigma2=sigma2_true)
        map_a, cov_a = low_rank_laplace_posterior_knownvar(Xa, ya, beta, sigma2_assumed, v0, rank=rank)
        map_b, cov_b = low_rank_laplace_posterior_knownvar(Xb, yb, beta, sigma2_assumed, v0, rank=rank)
        ds[m] = nbu.minus2logBC_mvn(map_a, cov_a, map_b, cov_b)
    return float(ds.mean()), float(ds.std(ddof=1) / np.sqrt(M)), ds
