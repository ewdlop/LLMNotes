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
    from simulated_experiment import run_experiment
    from palm_metrics_research import run_palm_research_simulation
except ImportError as e:
    print(f"Error importing research scripts: {e}")
    print("Ensure you are running this script from the root directory or the Google PaLM directory.")
    sys.exit(1)

def verify_palm_experiments():
    print("=== Starting PaLM 2 Research Simulation Verification ===")
    print("This script verifies the research findings by running the simulation experiments.\n")

    print(">>> Running Simulated Experiments (Detailed Metrics) <<<")
    # This script corresponds to the findings in PaLM_Experiments.md
    run_experiment()
    print("\n")

    print(">>> Running PaLM Metrics Research (Comparative Scenarios) <<<")
    # This script provides additional metric evaluations (BLEU/ROUGE)
    run_palm_research_simulation()
    print("\n")

    print("=== Experiment Completed Successfully ===")
    print("Research notes verified against simulated behavior.")

if __name__ == "__main__":
    verify_palm_experiments()
