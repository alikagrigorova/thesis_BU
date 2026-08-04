# Well-Calibrated Posteriors Under Model Misspecification via β-Bayes

This repository contains my bachelor's thesis. The complete thesis is in [`thesis_AlikaGrigorova.pdf`](thesis_AlikaGrigorova.pdf).

## Problem

My thesis identifies and solves the following problem in Bayesian statistics: imagine the model (likelihood) one is using for inference does not correctly represent reality. In practice, that is the case almost always, as real-world data-generating processes are more complex than a standard distribution can capture. It has been observed that under model misspecification, the following problem holds: if we compute posteriors using independent data realizations of the same data-generating process, we might get posteriors with essentially no overlap — which can lead to contradictory conclusions.

My thesis identifies the correct amount of overlap between posteriors and suggests a way of recovering it using Generalized Bayesian Inference with β-divergence loss.

## Ongoing work

After the thesis defense, I continue this project under the supervision of Professor Jonathan Huggins at Boston University. The current focus is extending the framework to high-dimensional settings.
