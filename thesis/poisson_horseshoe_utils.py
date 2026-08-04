"""Poisson regression + regularized horseshoe prior: beta-Bayes calibration experiment.

Implements the design in poisson_horseshoe_experiment_spec.md: a non-conjugate,
non-Gaussian extension of the beta-Bayes calibration framework (well-specified
threshold tau*, observed discrepancy d_obs, root-find beta*) used elsewhere in the
thesis for Gaussian/NIG models. Uses NumPyro (JAX) for NUTS sampling of both the
standard-Bayes (beta=1) and beta-Bayes (beta>1) horseshoe posteriors, and a
two-directional bridge estimator for the Bhattacharyya coefficient between two
posteriors of the same type (since neither has a closed form here).
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import jax

# Enable float64 globally: the beta-divergence loss exponentiates log-densities
# (potentially large in magnitude, e.g. gammaln over a 201-term truncated sum) and
# then sums/subtracts them, which is numerically delicate in float32. NUTS's
# gradient-based sampling also benefits from the extra precision generally.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random
from jax.scipy.special import gammaln, logsumexp
from scipy.special import gammaln as gammaln_np, logsumexp as logsumexp_np

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_density


Array = np.ndarray


def _log(msg: str, log_file: Optional[str] = None) -> None:
    """Print msg, and if log_file is given, also append it there immediately
    (flushed) -- lets a long-running fit's progress be tailed live from another
    process/terminal, independent of whatever buffers stdout (e.g. nbclient,
    which only writes a notebook's cell outputs to disk once the whole notebook
    finishes, not incrementally per cell).
    """
    print(msg, flush=True)
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(msg + "\n")


def load_or_compute_json(
    checkpoint_path: str, compute_fn: Callable[[], dict], log_file: Optional[str] = None,
    label: str = "",
) -> dict:
    """Generic checkpoint: if checkpoint_path already exists on disk, load and
    return its JSON contents (skipping computation entirely). Otherwise call
    compute_fn() (must return a JSON-serializable dict), save the result to
    checkpoint_path, and return it.

    This makes every expensive stage compute at most once, ever -- even across
    killing/restarting the notebook process for unrelated reasons (e.g. to add a
    new cell downstream). No result value is ever hardcoded into notebook code;
    the notebook always asks "has this exact computation already been saved?"
    rather than a human transcribing a remembered number.
    """
    import json
    import os

    prefix = f"[{label}] " if label else ""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            result = json.load(f)
        _log(f"{prefix}Loaded cached result from {checkpoint_path} (skipping computation).", log_file)
        return result

    result = compute_fn()
    with open(checkpoint_path, "w") as f:
        json.dump(result, f, indent=2)
    _log(f"{prefix}Saved result to {checkpoint_path}.", log_file)
    return result


def load_or_compute_pickle(
    checkpoint_path: str, compute_fn: Callable[[], object], log_file: Optional[str] = None,
    label: str = "",
) -> object:
    """Same idea as load_or_compute_json, but via pickle instead of JSON -- for
    results containing raw numpy/jax arrays (e.g. posterior samples) that JSON
    can't serialize directly. Used to persist actual fitted posteriors to disk,
    not just their derived summary statistics, so they never need refitting to
    compute some new downstream metric later."""
    import os
    import pickle

    prefix = f"[{label}] " if label else ""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            result = pickle.load(f)
        _log(f"{prefix}Loaded cached posteriors from {checkpoint_path} (skipping computation).", log_file)
        return result

    result = compute_fn()
    with open(checkpoint_path, "wb") as f:
        pickle.dump(result, f)
    _log(f"{prefix}Saved posteriors to {checkpoint_path}.", log_file)
    return result


def load_or_fit_blocks(
    checkpoint_path: str, model_fn: Callable, blocks: list, num_warmup: int = 500,
    num_samples: int = 500, base_seed_mcmc: int = 0, target_accept_prob: float = 0.8,
    max_tree_depth: int = 10, log_file: Optional[str] = None, label: str = "",
) -> list:
    """fit_blocks, wrapped with disk-backed persistence of the actual posterior
    samples (via load_or_compute_pickle) -- so every block-level fit anywhere in
    the pipeline (d_obs_std, beta* search evaluations, Tables 4/5/6) is computed
    at most once, ever, and the real fitted posteriors stay available on disk for
    any future analysis, not just whatever summary statistic was computed first.
    """
    def _compute():
        return fit_blocks(model_fn, blocks, num_warmup=num_warmup, num_samples=num_samples,
                           base_seed_mcmc=base_seed_mcmc, target_accept_prob=target_accept_prob,
                           max_tree_depth=max_tree_depth, log_file=log_file, label=label)

    return load_or_compute_pickle(checkpoint_path, _compute, log_file, label)


# ==============================
# Data-generating processes
# ==============================


def true_mu(X: Array, b0_true: float, b_true: Array) -> Array:
    """Poisson mean function mu_i = exp(b0 + x_i^T b) for the fixed true signal.

    NOTE: earlier versions of this function clipped the linear predictor (to keep
    mu bounded for the beta-loss's truncated sum, see beta_loss_poisson). That
    clip was a mistake here: it silently truncates ~9% of the "true" DGP itself
    (verified empirically -- at b_true=[2,-1.5,1,0..], eta exceeds 4.0 for ~9% of
    observations, exceeds 5.0 for ~4%), and neither numpy's Poisson/NegBin
    samplers nor NumPyro's built-in Poisson.log_prob need mu to be bounded at all.
    The truncated sum used ONLY for beta>1 (beta_loss_poisson) is the one place
    that genuinely needs a bound, and that's handled there via a generous k_trunc
    instead, so there is no clipping anywhere in the DGP or the standard-Bayes path.
    """
    eta = b0_true + X @ np.asarray(b_true)
    return np.exp(eta)


def generate_poisson_data(
    n: int, p: int, b0_true: float, b_true: Array, seed: int
) -> Tuple[Array, Array]:
    """Simulate (X, y) from the model's OWN assumed likelihood (Poisson) at the true
    coefficients. Used for the well-specified threshold tau* (Q = point mass at the
    true params, Definition 1), NOT for the observed-discrepancy computations.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    mu = true_mu(X, b0_true, b_true)
    y = rng.poisson(mu)
    return X, y


def generate_negbin_data(
    n: int, p: int, b0_true: float, b_true: Array, r: float, seed: int
) -> Tuple[Array, Array]:
    """Simulate (X, y) from the TRUE (misspecified) data-generating process: a
    Negative-Binomial with the same mean mu_i as the Poisson case but overdispersed,
    Var(y_i) = mu_i + mu_i^2/r, using the mean/size parameterization (not (n, p)).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    mu = true_mu(X, b0_true, b_true)
    # Convert mean/size (mu, r) -> classic (n=r, p=r/(r+mu)) parameterization for
    # np.random.negative_binomial, which takes (n successes, p success-prob).
    p_nb = r / (r + mu)
    y = rng.negative_binomial(r, p_nb)
    return X, y


# ==============================
# Regularized (Piironen-Vehtari) horseshoe, non-centered, + beta-divergence loss
# ==============================


def tau0_default(p: int, p0: int, n: int) -> float:
    """Standard Piironen & Vehtari recipe for the global-scale hyperparameter,
    informed by a prior guess p0 at the number of nonzero coefficients."""
    return (p0 / (p - p0)) * (1.0 / np.sqrt(n))


def poisson_logpmf(k: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
    """log Poisson(k; mu), vectorized. k and mu broadcast against each other."""
    return k * jnp.log(mu) - mu - gammaln(k + 1.0)


def beta_loss_poisson(
    y: jnp.ndarray, mu: jnp.ndarray, beta: float, k_trunc: int = 5000
) -> jnp.ndarray:
    """Sum over observations of the pointwise beta-divergence loss

        ell_beta(y_i, theta) = -1/(beta-1) f(y_i;theta)^(beta-1)
                                + 1/beta sum_{k=0}^{K} f(k;theta)^beta

    for a Poisson likelihood f(.|mu_i), beta > 1. K is fixed at k_trunc (a static
    Python int, required so shapes stay static under jit/vmap) rather than adapted
    per-observation to mu_i as in the original spec.

    k_trunc=5000 is deliberately generous rather than tightly matched to the DGP's
    expected mu range: an earlier version clipped mu (via the linear predictor) to
    keep it within a much smaller k_trunc=200's reach, which silently biased
    inference (verified empirically -- posterior means for the true nonzero
    coefficients stayed persistently below their true values even as n grew from
    100 to 2000, because clipping suppressed the likelihood's gradient signal for
    any observation whose predictor exceeded the clip). Checked directly: this
    truncated sum converges to the same value at k_trunc=5000 vs 10000 for mu up
    to 3000, and at k_trunc=60000 vs 100000 for mu up to 30000, and costs well
    under a second even at k_trunc=100000 -- so a large k_trunc with NO clip
    anywhere is both correct and computationally cheap here. Returns a scalar:
    sum_i ell_beta(y_i, theta).
    """
    beta = float(beta)
    log_fy = poisson_logpmf(y, mu)  # (n,)
    term1 = -(1.0 / (beta - 1.0)) * jnp.exp((beta - 1.0) * log_fy)

    k_grid = jnp.arange(k_trunc + 1, dtype=jnp.float64)  # (K+1,)
    # log f(k; mu_i)^beta = beta * (k log mu_i - mu_i - lgamma(k+1)), for every (i, k)
    log_terms = beta * (
        k_grid[None, :] * jnp.log(mu)[:, None]
        - mu[:, None]
        - gammaln(k_grid + 1.0)[None, :]
    )  # (n, K+1)
    log_sum_k = logsumexp(log_terms, axis=1)  # (n,) = log sum_k f(k;mu_i)^beta
    term2 = (1.0 / beta) * jnp.exp(log_sum_k)

    ell = term1 + term2  # (n,)
    return jnp.sum(ell)


def _poisson_likelihood_factor(
    X: jnp.ndarray,
    y: Optional[jnp.ndarray],
    b0: jnp.ndarray,
    b: jnp.ndarray,
    beta: float,
    k_trunc: int,
) -> None:
    """Shared likelihood/beta-loss logic for any prior on (b0, b): no clipping on
    the linear predictor or mu anywhere. beta=1 uses NumPyro's exact Poisson
    log_prob (no truncation, handles arbitrarily large mu directly). beta>1 uses
    the beta-divergence loss's truncated sum (beta_loss_poisson), which needs
    k_trunc large enough to cover whatever mu the sampler explores -- see that
    function's docstring; k_trunc is generous (default 5000) specifically so no
    additional clip on mu is needed here.
    """
    eta = b0 + X @ b
    mu = jnp.exp(eta)

    if beta == 1.0:
        numpyro.sample("obs", dist.Poisson(mu), obs=y)
    else:
        neg_loss = -beta_loss_poisson(y, mu, beta, k_trunc=k_trunc)
        numpyro.factor("beta_loglik", neg_loss)


def horseshoe_poisson_model(
    X: jnp.ndarray,
    y: Optional[jnp.ndarray],
    beta: float = 1.0,
    tau0: float = 1.0,
    slab_scale: float = 2.0,
    k_trunc: int = 5000,
) -> None:
    """Poisson regression with a non-centered regularized horseshoe prior.

    beta=1.0 uses the exact standard-Bayes Poisson likelihood (numpyro.sample with
    obs=y). beta>1.0 uses the beta-divergence loss as a numpyro.factor instead,
    recovering the beta-Bayes generalized posterior (thesis eq. 3/5).
    """
    n, p = X.shape

    tau = numpyro.sample("tau", dist.HalfCauchy(scale=tau0))
    lam = numpyro.sample("lam", dist.HalfCauchy(scale=jnp.ones(p)))
    c2 = numpyro.sample("c2", dist.InverseGamma(concentration=2.0, rate=slab_scale**2))

    lam_tilde = jnp.sqrt(c2 * lam**2 / (c2 + tau**2 * lam**2))
    b_raw = numpyro.sample("b_raw", dist.Normal(0.0, 1.0).expand([p]).to_event(1))
    b = numpyro.deterministic("b", tau * lam_tilde * b_raw)
    b0 = numpyro.sample("b0", dist.Normal(0.0, 5.0))

    _poisson_likelihood_factor(X, y, b0, b, beta, k_trunc)


def laplace_poisson_model(
    X: jnp.ndarray,
    y: Optional[jnp.ndarray],
    beta: float = 1.0,
    b_scale: float = 1.0,
    k_trunc: int = 5000,
) -> None:
    """Poisson regression with an iid Laplace (Bayesian Lasso) prior on b, in place
    of the horseshoe. Diagnostic variant to test whether the degenerate bridge-BC
    estimate seen with horseshoe is specifically due to its non-centered
    reparametrization scaffolding (tau, lambda_j, c2, b_raw_j) rather than a more
    fundamental issue with the full-latent-space bridge estimator itself: this
    model's ENTIRE latent space is just (b0, b) -- p+1 dimensions, no auxiliary
    variables to exclude -- so the "full space" and "(b0,b)-only" comparisons
    coincide here by construction.

    b0 unpenalized (same weak Normal(0,5) prior as the horseshoe model, for a
    like-for-like comparison); b_j ~ iid Laplace(0, b_scale).
    """
    n, p = X.shape

    b = numpyro.sample("b", dist.Laplace(0.0, b_scale).expand([p]).to_event(1))
    b0 = numpyro.sample("b0", dist.Normal(0.0, 5.0))

    _poisson_likelihood_factor(X, y, b0, b, beta, k_trunc)


# ==============================
# Batched NUTS fitting across replicate datasets
# ==============================


def fit_batch_nuts(
    model_fn: Callable,
    X_batch: jnp.ndarray,
    y_batch: jnp.ndarray,
    rng_key: jnp.ndarray,
    num_warmup: int = 500,
    num_samples: int = 500,
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
    """Fit model_fn independently on each of R datasets via jax.vmap over a single
    NUTS run per dataset (rather than looping mcmc.run() R times in serial).

    X_batch: (R, n, p), y_batch: (R, n). Returns (samples, diverging) where every
    value in `samples` has leading shape (R, num_samples, ...) and `diverging` has
    shape (R, num_samples). (Full R-hat needs >=2 chains per fit, which we skip here
    for speed -- divergence rate is the primary health check used below.)

    target_accept_prob/max_tree_depth default to NumPyro's own defaults (0.8, 10),
    preserving the synthetic-experiment results already computed with them. For the
    real-data (crime) notebook, target_accept_prob=0.99/max_tree_depth=15 (matching
    Bag_code's own Stan adapt_delta=0.99/max_treedepth=15) is needed -- verified
    empirically to eliminate a real convergence problem (one real-data block went
    from 100% divergent at default settings to 0% at these values).
    """
    R = X_batch.shape[0]
    rng_keys = random.split(rng_key, R)

    def run_one(key, X, y):
        kernel = NUTS(model_fn, target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                    num_chains=1, progress_bar=False)
        mcmc.run(key, X=X, y=y)
        samples = mcmc.get_samples()
        extra = mcmc.get_extra_fields()
        return samples, extra["diverging"]

    samples, diverging = jax.vmap(run_one)(rng_keys, X_batch, y_batch)
    return samples, diverging


def divergence_summary(diverging: jnp.ndarray, threshold_frac: float = 0.01) -> Dict[str, object]:
    """Per-replicate and overall divergence rates; flags replicates exceeding
    threshold_frac (default 1%, per the spec's data-quality gate)."""
    per_replicate_rate = np.asarray(diverging).mean(axis=1)
    flagged = np.where(per_replicate_rate > threshold_frac)[0]
    return {
        "per_replicate_rate": per_replicate_rate,
        "flagged_replicates": flagged,
        "n_flagged": int(len(flagged)),
        "overall_rate": float(np.asarray(diverging).mean()),
    }


# ==============================
# Two-directional bridge estimator for the Bhattacharyya coefficient
# ==============================


def latent_sample_sites(model_fn: Callable, X: jnp.ndarray, y: jnp.ndarray, rng_key=None) -> list:
    """Names of the model's actual latent `numpyro.sample` sites (excluding
    `numpyro.deterministic` sites like horseshoe's 'b', and excluding the observed
    'obs' site). These -- and only these -- are what log_density's `params` dict
    needs values for.

    This is determined dynamically (by tracing the model once) rather than
    hardcoded, since which names are "deterministic vs. sampled" differs between
    model variants -- e.g. 'b' is deterministic under the horseshoe prior but a
    real sample site under the Laplace prior. Hardcoding a name like "exclude b"
    silently breaks for any model where that assumption doesn't hold.
    """
    from numpyro.handlers import seed, trace

    rng_key = random.PRNGKey(0) if rng_key is None else rng_key
    tr = trace(seed(model_fn, rng_key)).get_trace(X, y)
    return [
        name for name, site in tr.items()
        if site["type"] == "sample" and not site.get("is_observed", False)
    ]


def _log_density_batch(
    model_fn: Callable,
    X: jnp.ndarray,
    y: jnp.ndarray,
    samples: Dict[str, jnp.ndarray],
    site_names: Optional[list] = None,
) -> jnp.ndarray:
    """Vectorized log_density(model_fn, (X, y), {}, params) over every posterior
    sample in `samples` (leading axis = sample index). Returns shape (num_samples,).

    site_names restricts which keys of `samples` are passed as latent params (see
    latent_sample_sites); if None, it's inferred by tracing model_fn(X, y) once.
    """
    if site_names is None:
        site_names = latent_sample_sites(model_fn, X, y)
    params = {k: v for k, v in samples.items() if k in site_names}

    def _one(p):
        log_p, _ = log_density(model_fn, (X, y), {}, p)
        return log_p

    return jax.vmap(_one)(params)


def bridge_bc_estimate(
    model_fn: Callable,
    samples1: Dict[str, jnp.ndarray],
    samples2: Dict[str, jnp.ndarray],
    X1: jnp.ndarray,
    y1: jnp.ndarray,
    X2: jnp.ndarray,
    y2: jnp.ndarray,
    log_ratio_clip: float = 30.0,
) -> Tuple[float, float]:
    """Two-directional bridge estimator for BC(Pi1, Pi2), Pi1/Pi2 posteriors of the
    SAME type (both standard Bayes, or both beta-Bayes at the same beta), fit on
    independent datasets (X1,y1) and (X2,y2).

        A_hat = E_{theta~Pi1}[ sqrt(exp(log_target(theta;D2) - log_target(theta;D1))) ]
        B_hat = E_{theta~Pi2}[ sqrt(exp(log_target(theta;D1) - log_target(theta;D2))) ]
        BC_hat = sqrt(A_hat * B_hat)

    Returns (BC_hat, d_hat) where d_hat = -2 log BC_hat (Renyi-1/2 discrepancy).
    """
    site_names = latent_sample_sites(model_fn, X1, y1)
    log_p1_D1 = _log_density_batch(model_fn, X1, y1, samples1, site_names)
    log_p1_D2 = _log_density_batch(model_fn, X2, y2, samples1, site_names)
    log_p2_D1 = _log_density_batch(model_fn, X1, y1, samples2, site_names)
    log_p2_D2 = _log_density_batch(model_fn, X2, y2, samples2, site_names)

    log_ratio_1 = jnp.clip(log_p1_D2 - log_p1_D1, -log_ratio_clip, log_ratio_clip)
    log_ratio_2 = jnp.clip(log_p2_D1 - log_p2_D2, -log_ratio_clip, log_ratio_clip)

    A_hat = jnp.mean(jnp.exp(0.5 * log_ratio_1))
    B_hat = jnp.mean(jnp.exp(0.5 * log_ratio_2))
    bc_hat = jnp.sqrt(A_hat * B_hat)
    bc_hat = jnp.clip(bc_hat, 1e-300, 1.0)
    d_hat = -2.0 * jnp.log(bc_hat)
    return float(bc_hat), float(d_hat)


def make_model(beta: float, tau0: float, slab_scale: float = 2.0, k_trunc: int = 5000) -> Callable:
    """Bind beta/tau0/slab_scale/k_trunc into a horseshoe model(X, y) closure for MCMC."""

    def _model(X, y=None):
        horseshoe_poisson_model(X, y, beta=beta, tau0=tau0, slab_scale=slab_scale, k_trunc=k_trunc)

    return _model


def make_laplace_model(beta: float, b_scale: float = 1.0, k_trunc: int = 5000) -> Callable:
    """Bind beta/b_scale/k_trunc into a Laplace-prior model(X, y) closure for MCMC."""

    def _model(X, y=None):
        laplace_poisson_model(X, y, beta=beta, b_scale=b_scale, k_trunc=k_trunc)

    return _model


# ==============================
# Replicate-pair Monte Carlo: well-specified threshold / observed discrepancy
# ==============================


def generate_replicate_pairs(gen_fn: Callable, R: int, base_seed: int) -> list:
    """R independent (Xa,ya,Xb,yb) pairs via gen_fn(seed=...), seeds base_seed+2i and
    base_seed+2i+1. gen_fn is generate_poisson_data or generate_negbin_data with all
    args bound except `seed` (see functools.partial usage in the notebook)."""
    pairs = []
    for i in range(R):
        Xa, ya = gen_fn(seed=base_seed + 2 * i)
        Xb, yb = gen_fn(seed=base_seed + 2 * i + 1)
        pairs.append((Xa, ya, Xb, yb))
    return pairs


def effective_dimension(model_fn: Callable, X: Array, samples: Dict[str, Array],
                         p: int, n_subsample: int = 10) -> Array:
    """Piironen-Vehtari effective-dimension kappa_j, adapted to a Poisson-GLM (log
    link): replaces the Gaussian case's fixed n/sigma^2 with the Fisher-information
    analogue Sum_i mu_i x_ij^2 (mu_i = the GLM's IRLS weight for a log-link Poisson),
    evaluated at each posterior draw's own (b0,b). Returns m_eff = sum_j(1-kappa_j)
    for n_subsample evenly-spaced posterior draws (cheap enough to call per-fit; see
    thesis discussion of horseshoe's adaptive effective dimensionality).
    """
    tau_s = np.asarray(samples["tau"]); lam_s = np.asarray(samples["lam"])
    c2_s = np.asarray(samples["c2"]); b0_s = np.asarray(samples["b0"])
    b_s = np.asarray(samples["b"])
    S = tau_s.shape[0]
    sub_idx = np.linspace(0, S - 1, n_subsample, dtype=int)
    m_eff_vals = np.zeros(len(sub_idx))
    for i, s_idx in enumerate(sub_idx):
        eta = b0_s[s_idx] + X @ b_s[s_idx]
        mu = np.exp(eta)
        fisher_j = (mu[:, None] * X**2).sum(axis=0)
        lam_tilde2 = c2_s[s_idx] * lam_s[s_idx]**2 / (c2_s[s_idx] + tau_s[s_idx]**2 * lam_s[s_idx]**2)
        kappa_j = 1.0 / (1.0 + fisher_j * tau_s[s_idx]**2 * lam_tilde2)
        m_eff_vals[i] = p - kappa_j.sum()
    return m_eff_vals


def pairwise_discrepancy_mc(
    model_fn: Callable,
    pairs: list,
    num_warmup: int = 500,
    num_samples: int = 500,
    chunk_size: int = 10,
    base_seed_mcmc: int = 0,
    log_ratio_clip: float = 600.0,
    collect_meff: bool = False,
    p_dim: Optional[int] = None,
    verbose: bool = True,
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
    log_file: Optional[str] = None,
) -> Dict[str, object]:
    """Fit model_fn via batched NUTS on both sides of every (Xa,ya,Xb,yb) pair in
    `pairs`, chunked (chunk_size pairs per jax.vmap call), computing the bridge-BC
    discrepancy d for each pair. This is the Monte Carlo engine shared by the
    well-specified threshold (tau*, pairs from generate_poisson_data) and the
    observed discrepancy (d_obs, pairs from generate_negbin_data) computations --
    same function, only the input pairs and model_fn (beta=1 vs beta>1) differ.

    If collect_meff, also computes the Piironen-Vehtari effective-dimension m_eff
    (see effective_dimension) for a subsample of posterior draws per fit -- only
    meaningful for the horseshoe model on well-specified (Poisson) data.

    Returns a dict with: ds (per-pair discrepancies, length R), mean_d, std_d, se_d,
    div_rate, and (if collect_meff) m_eff_mean/m_eff_se/D_eff/tau_pred_from_meff.
    """
    R = len(pairs)
    ds = []
    m_eff_list = []
    div_rates = []
    t_start = time.time()

    for chunk_idx in range(0, R, chunk_size):
        chunk_pairs = pairs[chunk_idx: chunk_idx + chunk_size]
        cs = len(chunk_pairs)
        Xa_b = np.stack([pr[0] for pr in chunk_pairs])
        ya_b = np.stack([pr[1] for pr in chunk_pairs])
        Xb_b = np.stack([pr[2] for pr in chunk_pairs])
        yb_b = np.stack([pr[3] for pr in chunk_pairs])

        key1 = random.PRNGKey(base_seed_mcmc + 2 * chunk_idx)
        key2 = random.PRNGKey(base_seed_mcmc + 2 * chunk_idx + 1)
        samples1, div1 = fit_batch_nuts(model_fn, Xa_b, ya_b, key1,
                                         num_warmup=num_warmup, num_samples=num_samples,
                                         target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth)
        samples2, div2 = fit_batch_nuts(model_fn, Xb_b, yb_b, key2,
                                         num_warmup=num_warmup, num_samples=num_samples,
                                         target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth)

        for i in range(cs):
            s1_i = {k: v[i] for k, v in samples1.items()}
            s2_i = {k: v[i] for k, v in samples2.items()}
            _, d = bridge_bc_estimate(model_fn, s1_i, s2_i, Xa_b[i], ya_b[i], Xb_b[i], yb_b[i],
                                       log_ratio_clip=log_ratio_clip)
            ds.append(d)
            if collect_meff:
                m_eff_list.extend(effective_dimension(model_fn, Xa_b[i], s1_i, p_dim))
                m_eff_list.extend(effective_dimension(model_fn, Xb_b[i], s2_i, p_dim))

        div_rates.append(float(np.asarray(div1).mean()))
        div_rates.append(float(np.asarray(div2).mean()))

        if verbose:
            n_done = chunk_idx + cs
            elapsed = time.time() - t_start
            eta = elapsed / n_done * (R - n_done)
            _log(f"  [{n_done}/{R}] mean_d={np.mean(ds):.4f}  std={np.std(ds, ddof=1) if n_done > 1 else 0:.4f}  "
                 f"div_rate={np.mean(div_rates):.4f}  elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min", log_file)

    ds = np.asarray(ds)
    result = {
        "ds": ds, "R": R,
        "mean_d": float(ds.mean()), "std_d": float(ds.std(ddof=1)),
        "se_d": float(ds.std(ddof=1) / np.sqrt(len(ds))),
        "div_rate": float(np.mean(div_rates)),
        "elapsed_min": (time.time() - t_start) / 60,
    }
    if collect_meff:
        m_eff_arr = np.asarray(m_eff_list)
        D_eff = m_eff_arr.mean() + 1.0
        result.update({
            "m_eff_mean": float(m_eff_arr.mean()),
            "m_eff_se": float(m_eff_arr.std(ddof=1) / np.sqrt(len(m_eff_arr))),
            "D_eff": float(D_eff),
            "tau_pred_from_meff": float(D_eff / 2.0),
        })
    if verbose:
        _log(f"\nFINAL: mean_d={result['mean_d']:.4f}  std={result['std_d']:.4f}  "
             f"SE={result['se_d']:.4f}  div_rate={result['div_rate']:.4f}  "
             f"time={result['elapsed_min']:.1f}min", log_file)
    return result


def pairwise_discrepancy_serial(
    model_fn: Callable,
    pairs: list,
    num_warmup: int = 500,
    num_samples: int = 500,
    base_seed_mcmc: int = 0,
    log_ratio_clip: float = 600.0,
    collect_meff: bool = False,
    p_dim: Optional[int] = None,
    verbose: bool = True,
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
    log_file: Optional[str] = None,
    checkpoint_prefix: Optional[str] = None,
) -> Dict[str, object]:
    """Same statistical computation as pairwise_discrepancy_mc (mean/SE bridge-BC
    discrepancy over R independent pairs, optional m_eff), but fits each side of
    each pair via a single, independent NUTS call (matching fit_blocks) instead of
    jax.vmap-batching multiple replicates together.

    Prefer this over pairwise_discrepancy_mc when replicates are likely to need
    very different NUTS trajectory lengths (e.g. real, messier data with a large
    max_tree_depth) -- vmap forces every replicate in a batch to share one
    computation graph, so if even one needs a deep tree, ALL of them pay that
    cost every iteration (the fast ones just sit through masked/wasted steps
    waiting for the slow one). A plain loop has no such coupling: one slow fit
    only costs its own time, not everyone else's. Verified empirically on the
    real crime dataset (p=100): a vmap-batched chunk of 10 pairs (20 fits) took
    76 min, vs ~9 min for a comparable 8-fit serial workload elsewhere in the
    same pipeline. Statistical correctness is identical either way -- NUTS's
    per-replicate stopping criterion and resulting samples don't depend on
    whether the fit ran inside a batch or alone, only wall-clock time does.
    """
    R = len(pairs)
    ds = []
    m_eff_list = []
    div_rates = []
    t_start = time.time()

    for i, (Xa, ya, Xb, yb) in enumerate(pairs):
        def _fit_side(key, X, y):
            kernel = NUTS(model_fn, target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth)
            mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=1, progress_bar=False)
            mcmc.run(key, X=jnp.asarray(X), y=jnp.asarray(y))
            div_rate = float(np.asarray(mcmc.get_extra_fields()["diverging"]).mean())
            return mcmc.get_samples(), div_rate

        if checkpoint_prefix is not None:
            s1, div1 = load_or_compute_pickle(
                f"{checkpoint_prefix}_pair{i}_a.pkl",
                lambda: _fit_side(random.PRNGKey(base_seed_mcmc + 2 * i), Xa, ya),
                log_file, label=f"pair {i+1}/{R} (a)")
            s2, div2 = load_or_compute_pickle(
                f"{checkpoint_prefix}_pair{i}_b.pkl",
                lambda: _fit_side(random.PRNGKey(base_seed_mcmc + 2 * i + 1), Xb, yb),
                log_file, label=f"pair {i+1}/{R} (b)")
        else:
            s1, div1 = _fit_side(random.PRNGKey(base_seed_mcmc + 2 * i), Xa, ya)
            s2, div2 = _fit_side(random.PRNGKey(base_seed_mcmc + 2 * i + 1), Xb, yb)

        _, d = bridge_bc_estimate(model_fn, s1, s2, jnp.asarray(Xa), jnp.asarray(ya),
                                   jnp.asarray(Xb), jnp.asarray(yb), log_ratio_clip=log_ratio_clip)
        ds.append(d)
        if collect_meff:
            m_eff_list.extend(effective_dimension(model_fn, Xa, s1, p_dim))
            m_eff_list.extend(effective_dimension(model_fn, Xb, s2, p_dim))
        div_rates.extend([div1, div2])

        if verbose:
            n_done = i + 1
            elapsed = time.time() - t_start
            eta = elapsed / n_done * (R - n_done)
            _log(f"  [{n_done}/{R}] d={d:.4f}  mean_d={np.mean(ds):.4f}  "
                 f"std={np.std(ds, ddof=1) if n_done > 1 else 0:.4f}  "
                 f"div_rate=({div1:.3f},{div2:.3f})  elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min", log_file)

    ds = np.asarray(ds)
    result = {
        "ds": ds, "R": R,
        "mean_d": float(ds.mean()), "std_d": float(ds.std(ddof=1)),
        "se_d": float(ds.std(ddof=1) / np.sqrt(len(ds))),
        "div_rate": float(np.mean(div_rates)),
        "elapsed_min": (time.time() - t_start) / 60,
    }
    if collect_meff:
        m_eff_arr = np.asarray(m_eff_list)
        D_eff = m_eff_arr.mean() + 1.0
        result.update({
            "m_eff_mean": float(m_eff_arr.mean()),
            "m_eff_se": float(m_eff_arr.std(ddof=1) / np.sqrt(len(m_eff_arr))),
            "D_eff": float(D_eff),
            "tau_pred_from_meff": float(D_eff / 2.0),
        })
    if verbose:
        _log(f"\nFINAL: mean_d={result['mean_d']:.4f}  std={result['std_d']:.4f}  "
             f"SE={result['se_d']:.4f}  div_rate={result['div_rate']:.4f}  "
             f"time={result['elapsed_min']:.1f}min", log_file)
    return result


# ==============================
# Real-data analog: within-dataset block U-statistic + posterior-based threshold
# (matches the Cal Housing / Appliances Energy methodology, Definition 1)
# ==============================


def make_k_blocks(X: Array, y: Array, K: int = 8, seed: int = 123) -> list:
    """Randomly split a real dataset (X, y) into K approximately-equal blocks."""
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    block_idx = np.array_split(perm, K)
    return [(X[idx], y[idx]) for idx in block_idx]


def fit_blocks(
    model_fn: Callable, blocks: list, num_warmup: int = 500, num_samples: int = 500,
    base_seed_mcmc: int = 0, target_accept_prob: float = 0.8, max_tree_depth: int = 10,
    log_file: Optional[str] = None, label: str = "",
) -> list:
    """Fit model_fn independently on each block via a plain loop (not jax.vmap --
    blocks may have unequal sizes since make_k_blocks uses np.array_split, and at
    this experiment's scale a single fit is fast enough, ~10-20s, that looping
    over K~8 blocks costs little). Returns [(samples, div_rate), ...] per block.

    target_accept_prob=0.99/max_tree_depth=15 (matching Bag_code's Stan
    adapt_delta=0.99/max_treedepth=15) is recommended for the real crime dataset --
    verified empirically to eliminate a real convergence problem at this dataset's
    scale/dimensionality (one block went from 100% divergent at NumPyro's own
    defaults to 0% at these values).
    """
    fits = []
    t_start = time.time()
    for i, (Xb, yb) in enumerate(blocks):
        kernel = NUTS(model_fn, target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                    num_chains=1, progress_bar=False)
        mcmc.run(random.PRNGKey(base_seed_mcmc + i), X=jnp.asarray(Xb), y=jnp.asarray(yb))
        samples = mcmc.get_samples()
        div_rate = float(np.asarray(mcmc.get_extra_fields()["diverging"]).mean())
        fits.append((samples, div_rate))
        elapsed = time.time() - t_start
        prefix = f"[{label}] " if label else ""
        _log(f"  {prefix}block {i+1}/{len(blocks)} fit  (div_rate={div_rate:.4f})  elapsed={elapsed/60:.1f}min",
             log_file)
    return fits


def block_u_statistic(
    model_fn: Callable, blocks: list, fits: list, log_ratio_clip: float = 600.0,
) -> Dict[str, object]:
    """Real-data analog of the observed discrepancy (d_obs): mean bridge-BC
    discrepancy across all C(K,2) block pairs, reusing the K independent
    per-block fits from fit_blocks (not R independent replicate pairs, since
    there is only one real dataset). SE via leave-one-block-out jackknife.
    """
    K = len(blocks)

    def _pairwise_mean(idx_subset):
        vals = []
        for a, i in enumerate(idx_subset):
            for j in idx_subset[a + 1:]:
                Xi, yi = blocks[i]; Xj, yj = blocks[j]
                si, _ = fits[i]; sj, _ = fits[j]
                _, d = bridge_bc_estimate(model_fn, si, sj, jnp.asarray(Xi), jnp.asarray(yi),
                                           jnp.asarray(Xj), jnp.asarray(yj),
                                           log_ratio_clip=log_ratio_clip)
                vals.append(d)
        return float(np.mean(vals)), vals

    full_mean, full_vals = _pairwise_mean(list(range(K)))

    loo_means = []
    for k in range(K):
        subset = [i for i in range(K) if i != k]
        m, _ = _pairwise_mean(subset)
        loo_means.append(m)
    loo_means = np.asarray(loo_means)
    var_jk = (K - 1.0) / K * np.sum((loo_means - loo_means.mean()) ** 2)
    se_jk = float(np.sqrt(max(var_jk, 0.0)))

    return {
        "mean_d": full_mean, "se_jk": se_jk,
        "pairwise_ds": np.asarray(full_vals),
        "div_rate": float(np.mean([f[1] for f in fits])),
    }


def generate_wellspec_pairs_from_posterior(
    ref_samples: Dict[str, jnp.ndarray], X_block: Array, R: int, base_seed: int = 0,
) -> list:
    """R independent (Xa,ya,Xb,yb) pairs for the well-specified threshold on real
    data (Q = posterior, Definition 1): sample theta_m=(b0,b) from a reference
    posterior (fit on the FULL real dataset via fit_blocks-style standard-Bayes
    NUTS), then simulate two independent synthetic Poisson datasets on the same
    X_block (matching the block size used in block_u_statistic) at theta_m.
    Feed the result directly into pairwise_discrepancy_mc.
    """
    b0_s = np.asarray(ref_samples["b0"])
    b_s = np.asarray(ref_samples["b"])
    S = b0_s.shape[0]
    idx = np.random.default_rng(base_seed).choice(S, size=R, replace=False)
    pairs = []
    for m, s_idx in enumerate(idx):
        mu_m = np.exp(b0_s[s_idx] + X_block @ b_s[s_idx])
        rng = np.random.default_rng(base_seed + 1000 + m)
        y1 = rng.poisson(mu_m)
        y2 = rng.poisson(mu_m)
        pairs.append((X_block, y1, X_block, y2))
    return pairs


# ==============================
# Downstream stability/accuracy tables (real data), matching Cal Housing's
# Tables 4-6: parameter-inference stability, predictive-inference stability,
# and leave-one-block-out predictive accuracy.
# ==============================


def minus2logBC_mvn(mu1: Array, Sigma1: Array, mu2: Array, Sigma2: Array) -> float:
    """Joint -2 log Bhattacharyya coefficient between two multivariate Gaussians
    (same formula as nig_beta_overlap_utils.minus2logBC_mvn, duplicated here so
    this module has no cross-file dependency)."""
    Sigma_bar = 0.5 * (Sigma1 + Sigma2)
    delta = mu1 - mu2
    _, ld1 = np.linalg.slogdet(Sigma1)
    _, ld2 = np.linalg.slogdet(Sigma2)
    _, ld_bar = np.linalg.slogdet(Sigma_bar)
    maha = float(delta @ np.linalg.solve(Sigma_bar, delta))
    return 0.25 * maha + ld_bar - 0.5 * (ld1 + ld2)


def _param_vector(samples: Dict[str, Array]) -> Array:
    """Stack (b0, b) posterior draws into a single (S, p+1) array, b0 first."""
    b0_s = np.asarray(samples["b0"]).reshape(-1, 1)
    b_s = np.asarray(samples["b"])
    return np.concatenate([b0_s, b_s], axis=1)


def parameter_stability_table(fits: list) -> Dict[str, float]:
    """Table 4 analog: mean L2 distance between posterior means of (b0,b) and mean
    Frobenius norm between posterior covariances of (b0,b), across all C(K,2)
    block pairs, for one already-fit model (fits = output of fit_blocks)."""
    K = len(fits)
    means = [_param_vector(s).mean(axis=0) for s, _ in fits]
    covs = [np.cov(_param_vector(s), rowvar=False) for s, _ in fits]
    l2, fro = [], []
    for i in range(K):
        for j in range(i + 1, K):
            l2.append(np.linalg.norm(means[i] - means[j]))
            fro.append(np.linalg.norm(covs[i] - covs[j], ord="fro"))
    return {"mean_l2": float(np.mean(l2)), "mean_frobenius": float(np.mean(fro))}


def predictive_stability_table(fits: list, X_star: Array, seed: int = 0) -> float:
    """Table 5 analog: mean d_1/2 = -2 log BC between the actual posterior
    PREDICTIVE distributions p(y*|x*, data) at a fixed covariate grid X_star,
    across all block pairs -- matching how Cal Housing's NIG predictive
    (nig_joint_predictive) is y* ~ N(X* mu, (b/d)(I + X* V X*')), i.e. parameter
    uncertainty PLUS observation noise, not just the mean function's uncertainty.

    For each posterior draw, pushes (b0, b) through mu_star = exp(b0 + X_star @ b)
    and then simulates y_star ~ Poisson(mu_star), giving S draws of an n_eval-dim
    count vector per block. Approximates each block's distribution over y_star as
    multivariate Gaussian and applies the closed-form MVN Bhattacharyya formula --
    a Monte Carlo analog of Cal Housing's Table 5, since Poisson counts have no
    closed-form joint predictive BC.
    """
    K = len(fits)
    rng = np.random.default_rng(seed)
    y_star_stats = []
    for samples, _ in fits:
        b0_s = np.asarray(samples["b0"])
        b_s = np.asarray(samples["b"])
        eta_star = b0_s[None, :] + X_star @ b_s.T  # (n_eval, S)
        mu_star = np.exp(eta_star).T  # (S, n_eval)
        # Clip only for this simulation step (never for fitting/means above); see
        # lobo_cv_predictive_metrics for the same numpy Poisson-generator limit.
        y_star = rng.poisson(np.clip(mu_star, None, 1e15))  # (S, n_eval)
        y_star_stats.append((y_star.mean(axis=0), np.cov(y_star, rowvar=False)))

    discs = []
    for i in range(K):
        for j in range(i + 1, K):
            mu1, Sigma1 = y_star_stats[i]
            mu2, Sigma2 = y_star_stats[j]
            discs.append(minus2logBC_mvn(mu1, Sigma1, mu2, Sigma2))
    return float(np.mean(discs))


def lobo_cv_predictive_metrics(
    model_fn: Callable, blocks: list, num_warmup: int = 500, num_samples: int = 500,
    base_seed_mcmc: int = 0, target_accept_prob: float = 0.99, max_tree_depth: int = 15,
    n_pred_draws: int = 200, log_file: Optional[str] = None, label: str = "",
    checkpoint_prefix: Optional[str] = None,
) -> Dict[str, float]:
    """Table 6 analog: leave-one-block-out CV predictive metrics (MLPD, RMSE,
    Cov90/95/99, CRPS), all via Monte Carlo since Poisson counts have no
    closed-form Student-t-style predictive (unlike the NIG case). For each held-
    out block, fits model_fn on the other K-1 blocks combined, then evaluates:

      MLPD: mean_i log( (1/S) sum_s Poisson_pmf(y_i; mu_s,i) )  (log-mean-exp)
      RMSE: sqrt(mean_i (y_i - mean_s[mu_s,i])^2)
      Cov90/95/99: fraction of y_i within the [alpha/2, 1-alpha/2] quantiles of
        posterior-PREDICTIVE COUNT draws (Poisson(mu_s,i) resampled n_pred_draws
        times per posterior sample, not just mu_s,i itself)
      CRPS: standard two-sample MC estimator using those same predictive draws
    """
    K = len(blocks)
    mlpd_list, rmse_list, cov90_list, cov95_list, cov99_list, crps_list = [], [], [], [], [], []

    for k in range(K):
        X_te, y_te = blocks[k]
        X_tr = np.vstack([blocks[j][0] for j in range(K) if j != k])
        y_tr = np.concatenate([blocks[j][1] for j in range(K) if j != k])

        def _fit_fold():
            kernel = NUTS(model_fn, target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth)
            mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=1, progress_bar=False)
            mcmc.run(random.PRNGKey(base_seed_mcmc + k), X=jnp.asarray(X_tr), y=jnp.asarray(y_tr))
            div_rate = float(np.asarray(mcmc.get_extra_fields()["diverging"]).mean())
            return mcmc.get_samples(), div_rate

        if checkpoint_prefix is not None:
            samples, _ = load_or_compute_pickle(
                f"{checkpoint_prefix}_fold{k}.pkl", _fit_fold, log_file, label=f"{label} fold {k+1}/{K}")
        else:
            samples, _ = _fit_fold()
        b0_s = np.asarray(samples["b0"])
        b_s = np.asarray(samples["b"])
        S = b0_s.shape[0]

        mu_si = np.exp(b0_s[None, :] + X_te @ b_s.T).T  # (S, n_te)

        log_pmf = y_te[None, :] * np.log(mu_si) - mu_si - gammaln_np(y_te[None, :] + 1.0)
        mlpd_i = logsumexp_np(log_pmf, axis=0) - np.log(S)
        mlpd_list.append(float(mlpd_i.mean()))

        pred_mean_i = mu_si.mean(axis=0)
        rmse_list.append(float(np.sqrt(np.mean((y_te - pred_mean_i) ** 2))))

        rng = np.random.default_rng(base_seed_mcmc + 5000 + k)
        s_idx = rng.integers(0, S, size=n_pred_draws)
        # Clip only for this simulation step (never for fitting/MLPD/RMSE above):
        # numpy's Poisson generator has an internal representational limit on lam
        # (raises "lam value too large" well below float64's own overflow point).
        # An occasional held-out point x posterior draw combination can produce an
        # astronomically large mu here (no clip anywhere in the actual model, by
        # design -- see true_mu's docstring); 1e15 is far beyond any plausible
        # count-data scale, so this only prevents a hard crash in pathological
        # tail cases without materially changing the simulated draws.
        mu_sim = np.clip(mu_si[s_idx], None, 1e15)
        Y_draws = rng.poisson(mu_sim)  # (n_pred_draws, n_te)

        for level, cov_list in [(0.90, cov90_list), (0.95, cov95_list), (0.99, cov99_list)]:
            alpha = 1 - level
            lo = np.quantile(Y_draws, alpha / 2, axis=0)
            hi = np.quantile(Y_draws, 1 - alpha / 2, axis=0)
            cov_list.append(float(np.mean((y_te >= lo) & (y_te <= hi))))

        Y1 = Y_draws[: n_pred_draws // 2]
        Y2 = Y_draws[n_pred_draws // 2:]
        term1 = np.mean(np.abs(Y1 - y_te[None, :]), axis=0)
        term2 = np.mean(np.abs(Y1[:, None, :] - Y2[None, :, :]), axis=(0, 1))
        crps_i = term1 - 0.5 * term2
        crps_list.append(float(crps_i.mean()))

        if log_file is not None:
            prefix = f"[{label}] " if label else ""
            _log(f"  {prefix}LOBO fold {k+1}/{K} done", log_file)

    return {
        "mlpd": float(np.mean(mlpd_list)),
        "rmse": float(np.mean(rmse_list)),
        "cov90": float(np.mean(cov90_list)),
        "cov95": float(np.mean(cov95_list)),
        "cov99": float(np.mean(cov99_list)),
        "crps": float(np.mean(crps_list)),
    }
