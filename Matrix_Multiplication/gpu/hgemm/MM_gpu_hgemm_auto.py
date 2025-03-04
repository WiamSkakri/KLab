import subprocess
import csv
import re
from time import sleep
def run_command(m, n, k):
    # Construct the command string with the given matrix dimensions
    command = f"./MM_hgemm {m} {n} {k}"
    try:
        # Run the command and capture its output
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        # Use regex to find and extract the average time in milliseconds from the output
        # Pattern matches "Average time: X.XX ms"
        match = re.search(r"Average time: ([\d.]+) ms", result.stdout)
        if match:
            # If the time was found, convert it to a float and return
            return float(match.group(1))
        else:
            # If the time wasn't found, print the actual output and return None
            print(f"Error: Couldn't extract time for {m}x{n}x{k}")
            print(f"Command output: {result.stdout}")
            return None
    except subprocess.SubprocessError as e:
        print(f"Error running command: {e}")
        return None
def main():
    # Name of the output CSV file
    csv_filename = "MM_gpu_hgemm_results.csv"
    # Open the CSV file in write mode
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header row
        csvwriter.writerow(['M', 'N', 'K', 'Average Time (ms)'])
        # Iterate over dimensions from 10 to 100 with step size 1
        for dim in range(10, 101, 10):
            # Run the matrix multiplication command and get the average time
            # Using same dimensions for M, N, and K
            avg_time = run_command(dim, dim, dim)
            if avg_time is not None:
                # If a valid time was returned, print the result and write it to the CSV
                print(f"Dimensions: {dim}x{dim}x{dim}, Average Time: {avg_time:.2f} ms")
                csvwriter.writerow([dim, dim, dim, avg_time])
                # Optional: add a small delay between runs to prevent system overload
                sleep(0.1)
    print(f"\nResults have been saved to {csv_filename}")
if __name__ == "__main__":
    main()