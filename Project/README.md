# Abstract

The study of optimal percolation in complex networks is a critical subject for understanding the resilience and vulnerability of complex systems, 
ranging from biological networks to infrastructure and social systems. Multiplex networks, characterized by multiple layers of interconnected networks with 
the same set of nodes, exhibit unique percolation properties that are not fully captured by traditional single-layer network models. 
This work aims to contribute to the study of targeted attack strategies in duplex networks and offers a more refined lens through which the impact of node 
removal can be assessed. Specifically, by employing modified versions of the Duplex Collective Influence (DCI) algorithm — DCI_harmonic and DCI_Katz — we aim 
to capture the effect of node removal not only in its immediate vicinity but also in its farthest-reaching neighborhood. These new strategies are then compared 
to the standard DCI version, as well as to other state-of-the-art generalizations of single-layer procedures, in the presence of non-zero inter-layer degree 
correlation and edge overlap. The analysis is performed by presenting the results of large-scale simulations over both synthetic and real-world duplex networks. 
Our findings offer a new perspective on the understanding of network dynamics, confirming that optimal strategies for network disruption or reinforcement are 
highly context-specific, varying significantly across different types of duplex networks. However, the promising results on the robustness of the DCI_harmonic 
procedure to variations in edge overlap probability open avenues for future research, particularly in exploring the applicability of these findings in more 
diverse and dynamic network configurations.

# Description

The code includes algorithms that allows to construct duplex networks having a tunable amount of interlayer degree correlations and tunable structural overlap.
Before running the code it is necessary to compile all C codes using the command `make`. Furthermore, ensure that the scripts written in Bash have execute permissions: `chmod +x filename.sh`. Then execute them as `./filename.sh`

The repository contains an example of a duplex network with two layers of N = 1000 nodes and M = 3000 edges. 

The script `script_for_increasing_correlation_and_overlap.sh` takes two layers specified in the command line and returns a set of duplex networks with different edge overlap probabilities and interlayer degree correlations. The different targeted attack strategies to evaluate the optimal percolation of duplex networks are implemented in `percolation_basefunction.py`. To evaluate the performance of the methods, run `main.py <Number of nodes> <layer_list.txt> <output_file_path>` where `layer_list.txt` is a file containing the list of files containing the two layers and `output_file_path` is the directory in which the results will be stored.

# Requirements

`NetworkX`: run `pip install networkx` in your command line interface to install NetworkX.

For more details refer to [NetworkX Documentation](https://networkx.org)
