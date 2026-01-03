import sys
import os
import numpy as np

# Add the root directory to sys.path to import LLMDistanceMetrics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LLMDistanceMetrics import LLMDistanceMetrics

def run_experiment():
    print("Initializing Google PaLM Research Experiment (Simulated)...")
    print("Objective: Evaluate hypothetical PaLM 2 performance using standard metrics.\n")

    metrics = LLMDistanceMetrics()

    # Define test cases based on PaLM 2 capabilities

    # Case 1: Reasoning (Math)
    print("--- Experiment 1: Reasoning (Math Problem) ---")
    reference_math = "To solve this, first subtract 5 from 15, which is 10. Then divide by 2, giving 5."

    # Simulated PaLM 2 output (strong reasoning)
    palm_math = "First, we take 15 minus 5 to get 10. Dividing that by 2 results in 5."

    # Simulated Baseline output (weaker reasoning)
    baseline_math = "15 minus 5 is 10. 10 divided by 2 is 5."

    print(f"Reference: {reference_math}")
    print(f"PaLM (Simulated): {palm_math}")
    print(f"Baseline: {baseline_math}")

    palm_bleu = metrics.compute_bleu(reference_math, palm_math)
    baseline_bleu = metrics.compute_bleu(reference_math, baseline_math)

    palm_rouge = metrics.compute_rouge(reference_math, palm_math)
    baseline_rouge = metrics.compute_rouge(reference_math, baseline_math)

    print(f"PaLM BLEU: {palm_bleu:.4f}")
    print(f"Baseline BLEU: {baseline_bleu:.4f}")
    print(f"PaLM ROUGE-L: {palm_rouge['rougeL']:.4f}")
    print(f"Baseline ROUGE-L: {baseline_rouge['rougeL']:.4f}")
    print("\n")

    # Case 2: Multilingual (Idiom Translation)
    print("--- Experiment 2: Multilingual (Idiom Translation) ---")
    # Idiom: "Break a leg" -> Spanish "Mucha mierda" (theatrical context) or "Rómpete una pierna" (literal)
    # Reference assumes a good cultural translation
    reference_trans = "Mucha mierda"

    # Simulated PaLM 2 (captures nuance)
    palm_trans = "Mucha mierda"

    # Simulated Baseline (literal)
    baseline_trans = "Rómpete una pierna"

    print(f"Reference: {reference_trans}")
    print(f"PaLM (Simulated): {palm_trans}")
    print(f"Baseline: {baseline_trans}")

    palm_bleu_trans = metrics.compute_bleu(reference_trans, palm_trans)
    baseline_bleu_trans = metrics.compute_bleu(reference_trans, baseline_trans)

    print(f"PaLM BLEU: {palm_bleu_trans:.4f}")
    print(f"Baseline BLEU: {baseline_bleu_trans:.4f}")
    print("\n")

    # Case 3: Code Generation (Python)
    print("--- Experiment 3: Code Generation (Python) ---")
    reference_code = "def factorial(n):\n    return 1 if n == 0 else n * factorial(n-1)"

    # Simulated PaLM 2 (concise, correct)
    palm_code = "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)"

    # Simulated Baseline (verbose or slightly different)
    baseline_code = "def fact(num):\n    res = 1\n    for i in range(1, num + 1):\n        res *= i\n    return res"

    print("Reference Code:\n" + reference_code)
    print("PaLM Code:\n" + palm_code)

    palm_bleu_code = metrics.compute_bleu(reference_code, palm_code)
    baseline_bleu_code = metrics.compute_bleu(reference_code, baseline_code)

    print(f"PaLM BLEU: {palm_bleu_code:.4f}")
    print(f"Baseline BLEU: {baseline_bleu_code:.4f}")

    print("\nExperiment Complete.")

if __name__ == "__main__":
    run_experiment()
