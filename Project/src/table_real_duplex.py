import os
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt

directory = '../output_files/real_world_duplex/'
columns = ['HDA', 'EMD1step', 'EMD', 'DCI', 'DCIz', 'DCI katz', 'DCI harmonic']
df = pd.DataFrame(columns=columns)
row_names = []

# Iterating over files in the directory
for filename in os.listdir(directory):
    if filename != '.DS_Store':
        row_names.append(filename)
        filename = filename.__add__('/output_file.txt')
        print(filename)
        if filename.endswith(".txt"):  # Assuming all files are .txt files
            file_path = os.path.join(directory, filename)
            
            with open(file_path, 'r') as file:
                content = file.read()
                data = ast.literal_eval(content)
                values = [item[1] for item in data]
                df.loc[len(df)] = values  # Add values as a new row in the DataFrame

# Setting new row names
df.index = row_names

print(df)

# Setting the font size
plt.rcParams.update({'font.size': 20})

# Setting the columns width
column_widths = np.repeat(0.05, len(df.columns))

# Plotting the DataFrame as a table
fig, ax = plt.subplots(figsize=(15, 4))  # Adjust the size as needed
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, colWidths=column_widths, loc='center')

plt.show()

# Saving the figure
# fig_path = '../figure/table_real_duplex.png'
# plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.05)
