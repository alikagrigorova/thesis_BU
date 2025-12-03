# Stability and Predictive Performance of General Bayesian Inference Methods  
### Intermediate Results Summary  
*(Based on `intermediate_results.pdf`)*

---

## Overview

This repository contains all code, experiments, and intermediate results for my Bachelor’s Honors Thesis on **robust Bayesian predictive inference**.  
The goal of the project is to compare classical and generalized Bayesian methods—NLL, GVI (β- and γ-divergence), PCUQ (MMD, CRPS), and BayesBag—and evaluate their:

- Predictive accuracy  
- Robustness under model misspecification  
- Reproducibility across data bootstraps  
- Stability with respect to variational family choice  

The primary experimental dataset is **California Housing Prices** (Nugent 2017).

---

## 1. Problem Setup

We consider **Gaussian linear regression**:

\[
y_i = x_i^\top w + \varepsilon_i,\quad \varepsilon_i \sim \mathcal N(0,\sigma^2)
\]

Preprocessing steps:

- Targets \(y_i\) are **log-transformed** (non-negativity).
- Features \(X\) are **standardized**.
- An intercept column is added to the design matrix.

### Data Splitting and Bootstrapping

- Train/test split: **4:1**
- For each method we evaluate **three bootstrap resamples** of the training data.
- Purpose: quantify **reproducibility** of inference under resampling.

---

## 2. Methods Evaluated

### 2.1 General Bayesian Inference (GVI)

We study the GVI objective:

\[
q^*(w,\sigma^2) = \arg\min_{q \in Q}
\mathbb{E}_q\left[\sum_{i=1}^n \ell(y_i,x_i,w,\sigma^2)\right]
+ \mathrm{KL}(q \,||\, \pi)
\]

#### Implemented losses

- **β-divergence** (β = 1.3)  
- **γ-divergence** (γ = 0.7)

#### Variational families

- **Normal–Inverse-Gamma (conjugate)**
- **Normal–LogNormal mean-field (non-conjugate)**

For β-divergence:
- Closed-form inference is available for the conjugate case.
- Black-box reparameterization gradients are used for the non-conjugate case.

For γ-divergence:
- Several parameterizations were tested.  
- A fully closed-form objective is theoretically available but numerically unstable due to constraints on \(V\).  
- Final experiments use the Kawashima–Fujisawa (2016) empirical approximation, optimized by sampling.

### 2.2 Classical NLL (Negative Log-Likelihood)

NLL is evaluated using the same two variational families:

- **Normal–Inverse-Gamma:** closed-form objective  
- **Normal–LogNormal mean-field:** closed-form objective  

---

## 3. PCUQ Methods

Two prediction-centric uncertainty quantification methods were implemented:

- **PCUQ–MMD**  
- **PCUQ–CRPS**

Both methods operate directly on predictive distributions and emphasize distributional fidelity rather than KL-based fitting.

---

## 4. BayesBag

BayesBag (Huggins & Miller 2024) stabilizes inference by **bootstrapping posterior samples** and averaging the resulting distributions.  
This produces a more reproducible predictive distribution, particularly under model misspecification.

---

## 5. Evaluation Metrics

Each method is evaluated using:

### 5.1 Predictive Distribution  
For conjugate NIG posteriors, the predictive distribution is a **Student-t**:

\[
p(y_*|x_*) = t_{2a^*}\left(x_*^\top \mu^*, \frac{b^*}{a^*}(1+x_*^\top V^* x_*)\right)
\]

For mean-field posteriors, predictions are sampled empirically.

### 5.2 NLPD (Negative Log Predictive Density)

Measures calibrated predictive probability assigned to true test values.

### 5.3 CRPS (Continuous Ranked Probability Score)

\[
\text{CRPS} = \frac1M \sum_{m=1}^M |s_m - y| -
\frac1{2M^2} \sum_{m=1}^M \sum_{k=1}^M |s_m - s_k|
\]

Evaluates both accuracy and concentration of predictive samples.

### 5.4 Residual Plots and Posterior Visualizations

For each method and bootstrap we include:

- Posterior samples  
- Predictive mean plots  
- Residual-vs-true scatterplots for 100 test points  

(See `Figures/` folder.)

---

## 6. Summary of Current Results

### Normal–Inverse–Gamma (Conjugate)

| Method | NLPD | CRPS |
|--------|-------|-------|
| NLL   | 0.9105 | 0.4050 |
| BETA  | **0.8853** | **0.3822** |
| GAMMA | 1.1662 | 0.4460 |

### Mean-Field Variational Family

| Method | NLPD | CRPS |
|--------|-------|-------|
| BETA  | **0.8820** | **0.3822** |
| GAMMA | 8301.75 | 1.1593 |
| NLL   | 36756.55 | 1.2452 |

**Key observations:**

- β-divergence is consistently the best performer across both variational families.  
- Mean-field γ-divergence and NLL produce extreme numerical instability.  
- Conjugate models are dramatically more stable.  
- Residual plots across bootstraps show strong reproducibility for β-divergence and BayesBag.

---

## 7. Next Steps

- Formalize reproducibility metrics using posterior and predictive distances.  
- Conduct sensitivity tests under **prior perturbations** and **model misspecification**.  
- Extend experiments to **non-i.i.d. settings**.  
- Integrate PCUQ–CRPS visualizations and BayesBag predictive variance decomposition.  

---

## References

(See the PDF for the full list; includes Knoblauch et al. 2022, Fujisawa & Eguchi 2008, Kawashima & Fujisawa 2016, Shen et al. 2025, Huggins & Miller 2024.)

