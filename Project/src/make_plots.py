import matplotlib.pyplot as plt
import ast
import numpy as np

# read the files containing the outputs and extract the data
file_path_rho = '../output_files/output_file_rho_third_trial.txt'
file_path_overlap_02 = '../output_files/output_file_02_third_trial.txt'
file_path_overlap_04 = '../output_files/output_file_04_forth_trial.txt'
file_path_rewiring = '../output_files/rewiring_overlap/output_file.txt'

with open(file_path_rho, 'r') as file:
    outputs_rho = [dict(ast.literal_eval(line.strip())) for line in file.readlines()]

with open(file_path_overlap_02, 'r') as file:
    outputs_overlap_02 = [dict(ast.literal_eval(line.strip())) for line in file.readlines()]

with open(file_path_overlap_04, 'r') as file:
    outputs_overlap_04 = [dict(ast.literal_eval(line.strip())) for line in file.readlines()]

with open(file_path_rewiring, 'r') as file:
    outputs_rewiring = [dict(ast.literal_eval(line.strip())) for line in file.readlines()]

all_keys = list(outputs_rewiring[0].keys())
markers = ['p', '.', '*', '^', 'v', 'H', 'd', 'o']

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

for k in range(len(all_keys)):
    data = [out.get(all_keys[k])/1000 for out in outputs_rho]
    ax1.scatter(np.linspace(-1, 1, 21), data, marker = markers[k], label=all_keys[k])
    ax1.plot(np.linspace(-1, 1, 21), data, linestyle='-', alpha=0.3, linewidth=1)
ax1.set_title('Duplex with 0.00 edge overlap probability')
ax1.set_xlabel('Interlayer degree correlation')
ax1.set_ylabel('Relative size of the critical set')
ax1.legend()

for k in range(len(all_keys)):
    data = [out.get(all_keys[k])/1000 for out in outputs_overlap_02]
    ax2.scatter(np.linspace(-1, 1, 21), data, marker = markers[k], label=all_keys[k], legend=True)
    ax2.plot(np.linspace(-1, 1, 21), data, linestyle='-', alpha=0.3, linewidth=1)
ax2.set_title('Duplex with 0.20 edge overlap probability')
ax2.set_xlabel('Interlayer degree correlation')
ax2.set_ylabel('Relative size of the critical set')
ax2.legend()

for k in range(len(all_keys)):
    data = [out.get(all_keys[k])/1000 for out in outputs_overlap_04]
    ax3.scatter(np.linspace(-1, 1, 21), data, marker = markers[k], label=all_keys[k])
    ax3.plot(np.linspace(-1, 1, 21), data, linestyle='-', alpha=0.3, linewidth=1)
ax3.set_title('Duplex with 0.40 edge overlap probability')
ax3.set_xlabel('Interlayer degree correlation')
ax3.set_ylabel('Relative size of the critical set')
ax3.legend()

plt.tight_layout()
plt.savefig("../figures/interlyer_corr_vs_relative_size_critical_set_new.png")

plt.figure(figsize=(6, 6))
for k in range(len(all_keys)):
    data = [out.get(all_keys[k])/1000 for out in outputs_rewiring]
    plt.scatter(np.linspace(0.1, 0.9, 9), data, marker = markers[k], s = 100, label=all_keys[k])
    plt.plot(np.linspace(0.1, 0.9, 9), data, linestyle='-', alpha=0.3, linewidth=1)
plt.title('Duplex with ER layers')
plt.xlabel('Edge overlap')
plt.ylabel('Relative size of the critical set')
plt.legend()

plt.savefig("../figures/edge_overlap_vs_relative_size_critical_set_new.png")