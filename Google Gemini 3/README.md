# Google Gemini 3 Research

This directory contains research notes and simulated experiments related to **Google Gemini 3**, with a specific focus on its capabilities in **Bayesian Inference**.

## Contents

* `simulated_bayesian_experiment.py`: A Python script that simulates "Gemini 3" responses to Bayesian inference problems (e.g., updating priors, Bayes theorem) and compares them against a baseline and a ground truth reference using BLEU and ROUGE metrics.
* `verify_research.py`: Entry point to run the simulated experiments and verify the research findings.

## Bayesian Inference Experiments

The experiments focus on:
1.  **Beta-Binomial Updates:** Updating a prior belief based on observed data (e.g., coin flips).
2.  **Bayes' Theorem Application:** Solving medical diagnosis problems involving sensitivity, specificity, and prevalence.

## Usage

To run the experiments, execute `verify_research.py` from the root of the repository or from this directory.

```bash
python3 "Google Gemini 3/verify_research.py"
```
