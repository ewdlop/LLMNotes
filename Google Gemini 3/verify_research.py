import sys
import os

# Add the root directory to sys.path to import LLMDistanceMetrics if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add the current directory to sys.path to ensure we can import the other scripts
# even if running from the root
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from simulated_bayesian_experiment import run_experiment
except ImportError as e:
    print(f"Error importing research scripts: {e}")
    print("Ensure you are running this script from the root directory or the Google Gemini 3 directory.")
    sys.exit(1)

def verify_gemini_experiments():
    print("=== Starting Google Gemini 3 Bayesian Inference Verification ===")
    print("This script verifies the capability of Gemini 3 to handle Bayesian Inference tasks.\n")

    print(">>> Running Simulated Bayesian Experiments <<<")
    run_experiment()
    print("\n")

    print("=== Experiment Completed Successfully ===")

if __name__ == "__main__":
    verify_gemini_experiments()
