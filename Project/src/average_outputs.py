import json
import os

def read_and_average_outputs(overlaps, lists, iterations, output_dir):
    averaged_results = {}

    for overlap in overlaps:
        # Format overlap to have two decimal places only if it's not 0
        formatted_overlap = "{:.2f}".format(overlap) if overlap != 0 else str(overlap)
        for list_id in lists:
            # Initialize a dictionary to store the sum of values for each method
            method_sums = {}
            method_counts = {}

            # Read each iteration file and aggregate the values
            for iteration in iterations:
                file_path = f"{output_dir}/output_file_{formatted_overlap}/list_{list_id}/iteration_{iteration}.txt"
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    for method, value in data:
                        if method not in method_sums:
                            method_sums[method] = 0
                            method_counts[method] = 0
                        method_sums[method] += value
                        method_counts[method] += 1

            # Calculate averages and store them
            averaged_results[(formatted_overlap, list_id)] = {method: method_sums[method] / method_counts[method] for method in method_sums}

    return averaged_results

def save_averaged_results(averaged_results, output_file):
    with open(output_file, 'w') as file:
        for overlap_list_id, methods in averaged_results.items():
            file.write(f"Overlap: {overlap_list_id[0]}, List ID: {overlap_list_id[1]}\n")
            for method, average in methods.items():
                file.write(f"{method}: {average}\n")
            file.write("\n")


# Example usage
overlaps = [0, 0.20, 0.40]
lists = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
iterations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
output_dir = "../output_files"

averaged_outputs = read_and_average_outputs(overlaps, lists, iterations, output_dir)

# Specify the path for the output file
output_file = "../output_files/averaged_results.txt"
save_averaged_results(averaged_outputs, output_file)
