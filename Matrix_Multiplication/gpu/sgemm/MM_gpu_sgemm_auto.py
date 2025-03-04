import subprocess
import csv
import re
from time import sleep

def run_command(m, n, k):
    command = f"./MM_sgemm {m} {n} {k}"
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        match = re.search(r"Average execution time over 50 iterations: ([\d.]+) seconds", result.stdout)
        
        if match:
            return float(match.group(1)) * 1000  # Convert to milliseconds
        else:
            print(f"Error: Couldn't extract time for {m}x{n}x{k}")
            print(f"Command output: {result.stdout}")
            return None
    
    except subprocess.SubprocessError as e:
        print(f"Error running command: {e}")
        return None

def main():
    csv_filename = "MM_gpu_sgemm_results.csv"
    
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['M', 'N', 'K', 'Average Time (ms)'])
        
        for dim in range(10, 101, 10):
            avg_time = run_command(dim, dim, dim)
            if avg_time is not None:
                print(f"Dimensions: {dim}x{dim}x{dim}, Average Time: {avg_time:.2f} ms")
                csvwriter.writerow([dim, dim, dim, avg_time])
                sleep(0.1)
    
    print(f"\nResults have been saved to {csv_filename}")

if __name__ == "__main__":
    main()