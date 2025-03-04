import subprocess
import pandas as pd
import os
def run_matrix_multiplication(m, n, executable="./hemm_gpu"):
    """Run the matrix multiplication program with given dimensions."""
    try:
        result = subprocess.run(
            [executable, str(m), str(n)],
            capture_output=True,
            text=True,
            check=True
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None
def collect_results():
    """Collect results for different matrix dimensions."""
    results = []
    for m in range(10, 101, 10):
        n = m  # Using square matrices
        execution_time = run_matrix_multiplication(m, n)
        if execution_time is not None:
            results.append({
                'Dimension': m,
                'ExecutionTime_ms': execution_time
            })
    df = pd.DataFrame(results)
    df.to_csv('MM_hemm_gpu_results.csv', index=False)
if __name__ == "__main__":
    collect_results()