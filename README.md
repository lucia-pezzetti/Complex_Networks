# Complex_Networks

## Project
The project aims at studying the optimal percolation problem on a duplex networks composed of Erdos-Renyi layers. Specifically, by employing a modified version of the Duplex Collective Influence (DCI) algorithm, this new method offers a more refined lens through which the impact of node removal can be assessed. Unlike the standard version of DCI, this modified algorithm uses different centrality measures - the betweenness centrality, the Katz centrality and the harmonic centrality - so to be able to capture the effect of a node removal not only in its immediate vicinity, but also into its farther-away neighborhood.

This repository has been re-adapted from [Multiplex_optimal_percolation](https://github.com/andresantoro/Multiplex_optimal_percolation), the repository of the paper:

[A. Santoro & V. Nicosia "Optimal percolation in correlated multilayer networks with overlap",Phys. Rev. Research 2, 033122 (2020)](https://link.aps.org/doi/10.1103/PhysRevResearch.2.033122)

## Miniproject
This project aims at studying the Resolution limit in community detection and reports on the paper:

[Fortunato, S., & Barthelemy, M. (2007). Resolution limit in community detection. Proceedings of the National Academy of Sciences, 104(1), 36-41.]
(https://www.pnas.org/doi/10.1073/pnas.0605965104)

The Louvain Community Detection Algorithm is used to detect the best community partition of a given undirected network. Then, To tackle the issue of 
resolution limit, the analysis is constrained on each of the obtained module and the Louvain Community Detection Algorithm is re-applied. The underlying
idea is to study the optimal modularity of each of the modules to understand whether they are coherent groups or ensemble ofsmaller communities. 
This approach reveals the presence of a finer structure inside most of the detected communities.
