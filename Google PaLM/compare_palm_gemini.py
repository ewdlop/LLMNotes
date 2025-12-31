from typing import List, Dict
import random

class PalmComparison:
    """
    A conceptual comparison class to simulate how one would evaluate PaLM 2
    against its successor Gemini, given that PaLM 2 API is decommissioned.
    """

    def __init__(self):
        self.models = ["PaLM 2 (Legacy)", "Gemini (Active)"]
        self.capabilities = ["Text Generation", "Reasoning", "Multilingual", "Multimodal"]

    def simulate_response(self, model: str, prompt: str) -> str:
        """
        Simulates a response since we cannot call the actual PaLM API.
        """
        if model == "PaLM 2 (Legacy)":
            return f"[PaLM 2 Simulation]: Processing '{prompt}' using text-only pathways..."
        elif model == "Gemini (Active)":
            return f"[Gemini Simulation]: Processing '{prompt}' with native multimodal understanding..."
        return "Unknown Model"

    def compare_capabilities(self) -> Dict[str, Dict[str, str]]:
        """
        Returns a static comparison of capabilities.
        """
        return {
            "Text Generation": {
                "PaLM 2": "High quality, comparable to other 2023 SOTA models.",
                "Gemini": "Superior, with better nuance and longer context."
            },
            "Multimodal": {
                "PaLM 2": "Limited (mostly text-in/text-out).",
                "Gemini": "Native support for Images, Audio, Video, and Text."
            },
            "Reasoning": {
                "PaLM 2": "Strong in symbolic logic and math.",
                "Gemini": "State-of-the-art complex reasoning."
            }
        }

    def run_comparison_demo(self):
        print("=== PaLM 2 vs. Gemini Concept Demo ===\n")

        test_prompt = "Explain the difference between a list and a tuple in Python."

        print(f"Prompt: {test_prompt}\n")

        for model in self.models:
            response = self.simulate_response(model, test_prompt)
            print(f"--- {model} ---")
            print(response)
            print("")

        print("=== Capability Matrix ===")
        comparison = self.compare_capabilities()
        for capability, details in comparison.items():
            print(f"\n{capability}:")
            print(f"  PaLM 2: {details['PaLM 2']}")
            print(f"  Gemini: {details['Gemini']}")

if __name__ == "__main__":
    comparator = PalmComparison()
    comparator.run_comparison_demo()
