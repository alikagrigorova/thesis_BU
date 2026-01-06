# Stability and Predictive Performance of General Bayesian Inference Methods

This repository contains code and results for a Bachelor’s Honors Thesis (in progress) investigating robust Bayesian predictive inference. The project compares several classical and modern Bayesian inference methods on the California Housing Prices dataset. The main focus is on evaluating these methods' predictive accuracy, robustness to model misspecification, reproducibility with bootstrapped samples, and stability with respect to different model choices.

Experiments cover standard negative log-likelihood, generalized Bayesian inference with β- and γ-divergence, prediction-centric uncertainty quantification (PCUQ), and BayesBag. Both closed-form and variational approaches are applied, and results summarize each method's reliability and performance.

The file `Bayes_Overview` contains the overview of the Bayesian methodologies considered in the thesis. The file `intermediate_results_thesis` contains the description of the intermediate experiments and results. 
The files `california_housing_meanfield_VF.ipynb` and `california_housing_conjugate_VF.ipynb` include code for the case with the mean field variational family and conjugate variational family and prior respectively.
See the code and `Figures/` directory for more details. Full references are available in the accompanying thesis PDF.

