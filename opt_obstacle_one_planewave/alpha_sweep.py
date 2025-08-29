import subprocess
import numpy as np
import matplotlib.pyplot as plt
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
def run_with_alpha(alpha):
    try:
        result = subprocess.run(
            ['mpirun', '-n', '8', 'python', 'get_objective.py', str(alpha)],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per run
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        # Extract objective value from output
        for line in result.stdout.split('\n'):
            if line.startswith('FINAL_OBJECTIVE:'):
                return float(line.split(':')[1].strip())
        
        # If no FINAL_OBJECTIVE found, print debug info
        print(f"No objective found for alpha={alpha}")
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        return None
        
    except subprocess.TimeoutExpired:
        print(f"Timeout for alpha={alpha}")
        return None
    except Exception as e:
        print(f"Error with alpha={alpha}: {e}")
        return None

def main():
    # Define alpha values to test
    alphas = np.linspace(1, 100, 100)
    
    objectives = []
    successful_alphas = []
    
    print(f"Testing {len(alphas)} alpha values...")
    
    for i, alpha in enumerate(alphas):
        print(f"Progress: {i+1}/{len(alphas)} - Testing alpha = {alpha:.3f}", flush=True)
        
        obj = run_with_alpha(alpha)
        
        if obj is not None:
            objectives.append(obj)
            successful_alphas.append(alpha)
            print(f"  -> Objective: {obj:.6f}")
        else:
            print(f"  -> Failed")
    
    print(f"\nCompleted {len(successful_alphas)} successful runs out of {len(alphas)} total")
    
    if len(successful_alphas) == 0:
        print("No successful runs - cannot plot results")
        return
    
    # Save results to JSON
    results = {
        'alphas': successful_alphas,
        'objectives': objectives
    }
    
    with open('alpha_sweep_results_2.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to: alpha_sweep_results_2.json")
    
    # Create and save plot
    plt.figure(figsize=(12, 8))
    plt.plot(successful_alphas, objectives, 'b-o', markersize=4, linewidth=2)
    plt.xlabel('Step Size (Alpha)', fontsize=12)
    plt.ylabel('Objective Value', fontsize=12)
    plt.title('Objective Value vs Step Size', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add some statistics to the plot
    min_idx = np.argmin(objectives)
    optimal_alpha = successful_alphas[min_idx]
    min_objective = objectives[min_idx]
    
    plt.axvline(x=optimal_alpha, color='r', linestyle='--', alpha=0.7, label=f'Optimal α = {optimal_alpha:.3f}')
    plt.axhline(y=min_objective, color='r', linestyle='--', alpha=0.7)
    
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plot_filename = 'alpha_sweep_analysis2.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {plot_filename}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"Optimal alpha: {optimal_alpha:.6f}")
    print(f"Minimum objective: {min_objective:.6f}")
    print(f"Objective range: [{min(objectives):.6f}, {max(objectives):.6f}]")
    
    return results

if __name__ == "__main__":
    results = main()