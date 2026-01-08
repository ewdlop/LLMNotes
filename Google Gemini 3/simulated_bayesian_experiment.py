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

    # Case 3: The Monty Hall Problem
    print("--- Experiment 3: The Monty Hall Problem ---")
    print("Scenario: You pick Door 1. Host Monty Hall (who knows what's behind doors) opens Door 3, revealing a goat. Should you switch to Door 2?")

    reference_monty = "Bayes' Theorem confirms you should switch. Let H1 be the car is behind Door 1, H2 for Door 2, H3 for Door 3. Prior P(H1)=P(H2)=P(H3)=1/3. Evidence E is Host opens Door 3. P(E|H1)=1/2 (Host can open 2 or 3). P(E|H2)=1 (Host must open 3). P(E|H3)=0 (Host won't reveal car). Posterior P(H2|E) = P(E|H2)P(H2)/P(E) = (1 * 1/3) / (1/2*1/3 + 1*1/3 + 0) = (1/3) / (1/2) = 2/3. Switching doubles your chance."

    # Simulated Gemini 3 (Correct Bayesian derivation)
    gemini_monty = "Let C1, C2, C3 be the events that the car is behind Door 1, 2, or 3. P(C1)=P(C2)=P(C3)=1/3. You picked Door 1. The Host opens Door 3 (Event D3). We calculate P(C2|D3). Likelihoods: P(D3|C1) = 1/2 (Host picks randomly between 2 and 3). P(D3|C2) = 1 (Host must pick 3). P(D3|C3) = 0. Marginal P(D3) = 1/2 * 1/3 + 1 * 1/3 + 0 = 1/6 + 1/3 = 1/2. P(C2|D3) = (1 * 1/3) / (1/2) = 2/3. Conclusion: Switch to Door 2."

    # Simulated Baseline (Common intuition error)
    baseline_monty = "There are two doors left, Door 1 and Door 2. It's a 50/50 chance now, so it doesn't matter if you switch or stay."

    print(f"Reference: {reference_monty}")
    print(f"Gemini 3 (Simulated): {gemini_monty}")
    print(f"Baseline: {baseline_monty}")

    gemini_bleu_monty = metrics.compute_bleu(reference_monty, gemini_monty)
    baseline_bleu_monty = metrics.compute_bleu(reference_monty, baseline_monty)

    print(f"Gemini 3 BLEU: {gemini_bleu_monty:.4f}")
    print(f"Baseline BLEU: {baseline_bleu_monty:.4f}")
    print("\n")

    print("Experiment Complete.")

if __name__ == "__main__":
    run_experiment()
