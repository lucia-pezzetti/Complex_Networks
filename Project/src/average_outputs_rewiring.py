import json
import os

def read_and_average_outputs_rewiring(rewiring, iterations, output_dir):
    averaged_results = {}

    for rewiring_id in rewiring:
        # Initialize a dictionary to store the sum of values for each method
        method_sums = {}
        method_counts = {}

        # Read each iteration file and aggregate the values
        for iteration in iterations:
            file_path = f"{output_dir}/rewiring_overlap/output_file{rewiring_id}/iteration_{iteration}.txt"
            with open(file_path, 'r') as file:
                data = json.load(file)
                for method, value in data:
                    if method not in method_sums:
                        method_sums[method] = 0
                        method_counts[method] = 0
                    method_sums[method] += value
                    method_counts[method] += 1

            # Calculate averages and store them
            averaged_results[f'0.{rewiring_id}'] = {method: method_sums[method] / method_counts[method] for method in method_sums}

    return averaged_results

def save_averaged_results(averaged_results, output_file):
    with open(output_file, 'w') as file:
        for rewiring_id, methods in averaged_results.items():
            file.write(f"Rewiring: {rewiring_id} \n")
            for method, average in methods.items():
                file.write(f"{method}: {average}\n")
            file.write("\n")

# Specify the lists and iterations to be used
rewiring = [10, 20, 30, 40, 50, 60, 70, 80, 90]
iterations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
output_dir = "../output_files"

averaged_outputs = read_and_average_outputs_rewiring(rewiring, iterations, output_dir)

# Specify the path for the output file
output_file = "../output_files/averaged_results_rewiring.txt"
save_averaged_results(averaged_outputs, output_file)
