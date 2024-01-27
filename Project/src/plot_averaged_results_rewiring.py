import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# First, let's read the content of the uploaded file to understand its structure and format
file_path = '../output_files/averaged_results_rewiring.txt'

with open(file_path, 'r') as file:
    file_content = file.read()

# Creating a list for each line in the content
lines = file_content.split('\n')

# Adjusting the parsing to handle decimal values for overlap
data = {
    "Rewiring": [],
    "Method": [],
    "Value": [],
}

for line in lines:
    if line.startswith('Rewiring'):
        current_rewiring = float(line.split(':')[-1].strip())  # Parsing as float
    elif line:
        method, value = line.split(':')
        data["Rewiring"].append(current_rewiring)
        data["Method"].append(method.strip())
        data["Value"].append(float(value.strip())/1000)

# Creating a DataFrame
df = pd.DataFrame(data)

# Defining a set of markers for distinct visualization
markers = ['p', '.', '*', '^', 'v', 'H', 'd', 'o']

# Plotting the data
plt.figure(figsize=(6, 6))

# Creating subplots for each unique overlap

# Plotting scatter plot and connecting points of the same method with lines
for j, method in enumerate(df['Method'].unique()):
    method_subset = df[df['Method'] == method]
    sns.lineplot(x='Rewiring', y='Value', data=method_subset, lw=1)
    sns.scatterplot(x='Rewiring', y='Value', data=method_subset, s=100, marker=markers[j % len(markers)], label=method)
    #sns.scatterplot(x="Mapped List ID", y="Value", hue="Method", data=subset)
# plt.title(f'Duplex with {rewiring} edge overlap probability')
plt.xlabel('Edge overlap probability')
plt.ylabel('Relative size of the critical set')

plt.tight_layout()
plt.show()
