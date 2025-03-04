
import subprocess
import csv
import re
from time import sleep
def run_command(m, n, k):
    """
    Run the matrix multiplication benchmark with given dimensions
    and extract the execution time from the output.
    """
    # Construct the command string with the given matrix dimensions
    command = f"./MM_dgemm {m} {n} {k}"
    try:
        # Run the command and capture its output
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        # Extract the time value from the output
        # The original C++ code outputs: m,n,k,time
        # Split the output by commas and get the last value (time)
        if result.stdout.strip():
            m, n, k, time = result.stdout.strip().split(',')
            return float(time) * 1000  # Convert seconds to milliseconds
        else:
            print(f"Error: No output for dimensions {m}x{n}x{k}")
            print(f"Command output: {result.stdout}")
            return None
    except subprocess.SubprocessError as e:
        print(f"Error running command: {e}")
        return None
    except ValueError as e:
        print(f"Error parsing output for dimensions {m}x{n}x{k}: {e}")
        print(f"Command output: {result.stdout}")
        return None
def main():
    # Name of the output CSV file
    csv_filename = "matrix_multiplication_dgemm_results.csv"
    # Open the CSV file in write mode
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header row
        csvwriter.writerow(['M', 'N', 'K', 'Average Time (ms)'])
        # Iterate over dimensions from 10 to 100 with step size 10
        for dim in range(10, 101, 10):
            # Run the matrix multiplication command and get the average time
            # Using same dimensions for M, N, and K
            avg_time = run_command(dim, dim, dim)
            if avg_time is not None:
                # If a valid time was returned, print the result and write it to the CSV
                print(f"Dimensions: {dim}x{dim}x{dim}, Average Time: {avg_time:.2f} ms")
                csvwriter.writerow([dim, dim, dim, avg_time])
                # Add a small delay between runs to prevent system overload
                sleep(0.1)
    print(f"\nResults have been saved to {csv_filename}")
if __name__ == "__main__":
    main()