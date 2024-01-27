import sys
import networkx as nx
import numpy as np
from percolation_basefunctions import *
from optimal_percolation import *
import json
import os

if len(sys.argv) < 4:
    print("Usage: %s <Number of nodes> <layer_list> <output_file_path>"% sys.argv[0])
    sys.exit(1)

N = int(sys.argv[1])
multiplex_filename = sys.argv[2]
multiplex_structure = load_dataset_into_dict_of_dict(multiplex_filename,N)

#################    Multiplex targeted strategies (without Pareto)   #################

HDA_total_target_nodes,HDA_size_nodes,HDA_steps_attack = targeted_attack_adaptive(multiplex_structure,N,computing_product_degree)
print()
EMD1step_total_target_nodes,EMD1step_size_nodes,EMD1step_steps_attack = targeted_attack_adaptive(multiplex_structure,N,EMD1step_ranking)
print()
EMD_total_target_nodes,EMD_size_nodes,EMD_steps_attack = targeted_attack_adaptive(multiplex_structure,N,EMDfullsteps_ranking)
print()
DCI_total_target_nodes,DCI_size_nodes, DCI_steps_attack = targeted_attack_adaptive(multiplex_structure,N,compute_ranking_DCI)
print()
DCIz_total_target_nodes,DCIz_size_nodes, DCIz_steps_attack = targeted_attack_adaptive(multiplex_structure,N,compute_ranking_DCIz)
print()
# DCI_total_target_nodes_betweenness,DCI_size_nodes_betweenness, DCI_steps_attack_betweenness = targeted_attack_adaptive(multiplex_structure,N,compute_ranking_DCIz_betweenness_centrality)
# print()
DCI_total_target_nodes_katz,DCI_size_nodes_katz, DCI_steps_attack_katz = targeted_attack_adaptive(multiplex_structure,N,compute_ranking_DCI_katz_centrality)
print()
DCI_total_target_nodes_harmonic, DCI_size_nodes_harmonic, DCI_steps_attack_harmonic = targeted_attack_adaptive(multiplex_structure,N,compute_ranking_DCI_harmonic_centrality)
print()
# product_katz_total_target_nodes,product_katz_size_nodes, product_katz_steps_attack = targeted_attack_adaptive(multiplex_structure,N,compute_ranking_product_katz_centrality)
# print()
# product_harmonic_total_target_nodes, product_harmonic_size_nodes, product_harmonic_steps_attack = targeted_attack_adaptive(multiplex_structure,N,compute_ranking_product_harmonic_centrality)
# print()

###### Results
results_dict={
"HDA":HDA_size_nodes,
"EMD1step":EMD1step_size_nodes,
"EMD":EMD_size_nodes,
"DCI":DCI_size_nodes,
"DCIz":DCIz_size_nodes,
# "DCI betweenness":DCI_size_nodes_betweenness,
"DCI katz":DCI_size_nodes_katz,
"DCI harmonic":DCI_size_nodes_harmonic}
# "product katz":product_katz_size_nodes,
# "product harmonic":product_harmonic_size_nodes}

print(sorted(results_dict.items(),key= lambda x: x[1]))

# Extract the file path from the command line argument
file_path = sys.argv[3]

# Create the directory if it doesn't exist
dir_name = os.path.dirname(file_path)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# Write the data to the file
with open(file_path, 'a') as file:
    file.write(json.dumps(list(results_dict.items())) + '\n')