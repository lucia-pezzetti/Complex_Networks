import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# First, let's read the content of the uploaded file to understand its structure and format
file_path = '../output_files/averaged_results.txt'

with open(file_path, 'r') as file:
    file_content = file.read()

# Creating a list for each line in the content
lines = file_content.split('\n')

# Adjusting the parsing to handle decimal values for overlap
data = {
    "Overlap": [],
    "List ID": [],
    "Method": [],
    "Value": [],
}

for line in lines:
    if line.startswith('Overlap'):
        parts = line.split(',')
        current_overlap = float(parts[0].split(':')[-1].strip())  # Parsing as float
        current_list_id = int(parts[1].split(':')[-1].strip())
    elif line:
        method, value = line.split(':')
        data["Overlap"].append(current_overlap)
        data["List ID"].append(current_list_id)
        data["Method"].append(method.strip())
        data["Value"].append(float(value.strip())/1000)

# Creating a DataFrame
df = pd.DataFrame(data)

def map_list_id_to_range(list_id):
    return round(-1 + 0.1 * list_id, 1)

# Apply the mapping
df['Mapped List ID'] = df['List ID'].map(map_list_id_to_range)

# Defining a set of markers for distinct visualization
markers = ['p', '.', '*', '^', 'v', 'H', 'd', 'o']

# Plotting the data
plt.figure(figsize=(18, 6))

# Creating subplots for each unique overlap
unique_overlaps = df['Overlap'].unique()
for i, overlap in enumerate(unique_overlaps):
    plt.subplot(1, len(unique_overlaps), i + 1)
    subset = df[df['Overlap'] == overlap]
    # Plotting scatter plot and connecting points of the same method with lines
    for j, method in enumerate(subset['Method'].unique()):
        method_subset = subset[subset['Method'] == method]
        sns.lineplot(x='Mapped List ID', y='Value', data=method_subset, lw=1)
        sns.scatterplot(x='Mapped List ID', y='Value', data=method_subset, s=100, marker=markers[j % len(markers)], label=method)
    #sns.scatterplot(x="Mapped List ID", y="Value", hue="Method", data=subset)
    plt.title(f'Duplex with {overlap} edge overlap probability')
    plt.xlabel('Interlayer degree correlation')
    plt.ylabel('Relative size of the critical set')
    if i > 0:  # Hide the legend for subplots after the first one
        plt.legend([],[], frameon=False)

plt.tight_layout()
plt.show()
