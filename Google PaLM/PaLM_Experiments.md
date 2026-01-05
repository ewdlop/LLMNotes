# Google PaLM Research Experiment Notes

**Date:** October 2025
**Experiment:** Simulated Performance Evaluation of PaLM 2

## Context
Since the Google PaLM API has been decommissioned in favor of Gemini, direct inference testing is no longer possible. This research focuses on establishing an evaluation framework using the `LLMDistanceMetrics` library and simulated outputs based on PaLM 2's documented capabilities.

## Methodology
We define "Reference" texts representing ideal human outputs and "Simulated" outputs representing PaLM 2's behavior as described in its technical report (e.g., strong reasoning, idiom understanding, code correctness). We compare these against a "Baseline" (representing a weaker or more literal model).

## Results

### 1. Reasoning (Math)
*   **Scenario:** Solving a two-step math problem.
*   **Observation:** The simulated PaLM output, while semantically correct, received low BLEU scores due to phrasing differences. However, by using smoothing functions in BLEU calculation, we capture non-zero overlaps (BLEU ~0.035), indicating some structural similarity. ROUGE-L still provided a better measure of content overlap.

### 2. Multilingual (Idioms)
*   **Scenario:** Translating "Break a leg" to Spanish.
*   **Observation:** PaLM 2's documented ability to handle idioms ("Mucha mierda") matches the reference perfectly. With smoothing enabled, this perfect match yields a BLEU score of 0.3162 (limited by short length preventing higher-order n-grams), whereas the literal translation ("Rómpete una pierna") scores 0.0, correctly reflecting its inaccuracy.

### 3. Code Generation
*   **Scenario:** Python factorial function.
*   **Observation:** The simulated PaLM code (recursive, standard structure) achieved a high BLEU score (0.6503) against the reference, significantly outperforming the baseline iterative approach (BLEU 0.0215) in terms of similarity to the reference style.

## Conclusion
This framework demonstrates how `LLMDistanceMetrics` can be used to evaluate model outputs. Future work should involve replacing the simulated outputs with actual generations from the Gemini API (as the successor to PaLM) to track progress.
