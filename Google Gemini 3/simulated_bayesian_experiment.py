import sys
import os
import numpy as np

# Add the root directory to sys.path to import LLMDistanceMetrics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LLMDistanceMetrics import LLMDistanceMetrics

def run_experiment():
    print("Initializing Google Gemini 3 Research Experiment (Simulated)...")
    print("Objective: Evaluate hypothetical Gemini 3 performance on Bayesian Inference tasks.\n")

    metrics = LLMDistanceMetrics()

    # Case 1: Simple Bayesian Update (Coin Flip)
    print("--- Experiment 1: Simple Bayesian Update (Coin Flip) ---")
    print("Scenario: A coin is flipped 10 times, resulting in 7 Heads and 3 Tails. Prior is Beta(1,1).")

    reference_inference = "Given a Beta(1,1) prior (Uniform) and observing 7 Heads and 3 Tails, the posterior distribution is Beta(1+7, 1+3) = Beta(8, 4). The expected probability of Heads is 8/(8+4) = 8/12 = 0.6667."

    # Simulated Gemini 3 output (Accurate, detailed reasoning)
    gemini_inference = "Starting with a uniform Beta(1,1) prior. We update the hyperparameters with the observed data: alpha' = alpha + heads = 1 + 7 = 8, and beta' = beta + tails = 1 + 3 = 4. The posterior is Beta(8, 4). The mean of this distribution is 8 / (8 + 4) = 0.67."

    # Simulated Baseline output (Less precise or slightly confused)
    baseline_inference = "You saw 7 heads and 3 tails. So the probability is 70%. The prior doesn't matter much."

    print(f"Reference: {reference_inference}")
    print(f"Gemini 3 (Simulated): {gemini_inference}")
    print(f"Baseline: {baseline_inference}")

    gemini_bleu = metrics.compute_bleu(reference_inference, gemini_inference)
    baseline_bleu = metrics.compute_bleu(reference_inference, baseline_inference)

    print(f"Gemini 3 BLEU: {gemini_bleu:.4f}")
    print(f"Baseline BLEU: {baseline_bleu:.4f}")
    print("\n")

    # Case 2: Medical Diagnosis (Bayes Theorem)
    print("--- Experiment 2: Medical Diagnosis (Bayes Theorem) ---")
    print("Scenario: Disease D has prevalence 1%. Test T has 95% sensitivity (true positive) and 90% specificity (true negative). Patient tests positive.")

    # Calculation: P(D|T) = P(T|D)P(D) / (P(T|D)P(D) + P(T|~D)P(~D))
    # P(T|D) = 0.95, P(D) = 0.01
    # P(T|~D) = 1 - 0.90 = 0.10, P(~D) = 0.99
    # Num = 0.95 * 0.01 = 0.0095
    # Denom = 0.0095 + (0.10 * 0.99) = 0.0095 + 0.099 = 0.1085
    # Result = 0.0095 / 0.1085 = 0.0875... ~8.76%

    reference_diagnosis = "Using Bayes' theorem: P(D|+) = (P(+|D) * P(D)) / P(+). P(+) = P(+|D)P(D) + P(+|no D)P(no D) = 0.95*0.01 + 0.10*0.99 = 0.0095 + 0.099 = 0.1085. P(D|+) = 0.0095 / 0.1085 ≈ 0.0876 or 8.76%."

    # Simulated Gemini 3 (Step-by-step correct application)
    gemini_diagnosis = "We need to find P(Disease|Positive). P(Disease)=0.01. Sensitivity P(Positive|Disease)=0.95. False Positive Rate P(Positive|No Disease) = 1 - 0.90 = 0.10. Numerator = 0.95 * 0.01 = 0.0095. Denominator = 0.0095 + (0.10 * 0.99) = 0.1085. Probability is 0.0095 / 0.1085 ≈ 8.76%."

    # Simulated Baseline (Base rate fallacy)
    baseline_diagnosis = "The test is 95% accurate, so if you tested positive, there is a 95% chance you have the disease."

    print(f"Reference: {reference_diagnosis}")
    print(f"Gemini 3 (Simulated): {gemini_diagnosis}")
    print(f"Baseline: {baseline_diagnosis}")

    gemini_bleu_diag = metrics.compute_bleu(reference_diagnosis, gemini_diagnosis)
    baseline_bleu_diag = metrics.compute_bleu(reference_diagnosis, baseline_diagnosis)

    print(f"Gemini 3 BLEU: {gemini_bleu_diag:.4f}")
    print(f"Baseline BLEU: {baseline_bleu_diag:.4f}")
    print("\n")

    print("Experiment Complete.")

if __name__ == "__main__":
    run_experiment()
