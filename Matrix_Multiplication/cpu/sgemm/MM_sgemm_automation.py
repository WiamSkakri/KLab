import subprocess
import csv
import re
# Function to run the matrix multiplication command and extract the average execution time
def run_command(a, b, c):
    # Construct the command string with the given matrix dimensions
    command = f"./Matrix_multiply_sgemm {a} {b} {c}"
    # Run the command and capture its output
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # Use regex to find and extract the average execution time from the output
    match = re.search(r"Average execution time over 50 iterations: ([\d.]+) seconds", result.stdout)
    if match:
        # If the time was found, convert it to a float and return
        return float(match.group(1))
    else:
        # If the time wasn't found, print an error message and return None
        print(f"Error: Couldn't extract time for {a}x{b}x{c}")
        return None
# Name of the output CSV file
csv_filename = "matrix_multiply_results.csv"
# Open the CSV file in write mode
with open(csv_filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write the header row to the CSV file
    csvwriter.writerow(['Dimensions', 'Average Time (seconds)'])
    # Iterate over matrix dimensions from 10 to 100, incrementing by 10 each time
    for dim in range(10, 2000, 10):
        # Run the matrix multiplication command and get the average time
        avg_time = run_command(dim, dim, dim)
        if avg_time is not None:
            # If a valid time was returned, print the result and write it to the CSV
            print(f"Dimensions: {dim}x{dim}x{dim}, Average Time: {avg_time:.6f} seconds")
            csvwriter.writerow([f"{dim}x{dim}x{dim}", avg_time])
            # Print a message indicating where the results have been saved
print(f"Results have been saved to {csv_filename}")