# Degree centrlity fails to detect the importance of a node in a network

import networkx as nx
import matplotlib.pyplot as plt

# Create a new graph
G = nx.Graph()

# Add nodes
G.add_node("A", color='deepskyblue')
G.add_node("B", color='coral')
G.add_node("C", color='coral')

# Add edges for Node A
G.add_edge("A", "B")
G.add_edge("A", "C")

# Add additional edges to Nodes B and C to increase their degree
for i in range(1, 10):  # assuming a high degree is 10
    G.add_node(f"B{i}", color='navajowhite')
    G.add_node(f"C{i}", color='navajowhite')
    G.add_edge("B", f"B{i}")
    G.add_edge("C", f"C{i}")

# Get the color for each node
colors = [G.nodes[n]['color'] for n in G.nodes]

# Draw the graph
pos = nx.spring_layout(G)  # positions for all nodes
nx.draw(G, pos, node_color=colors, node_size=600, with_labels=True)
plt.show()

# Issue for the betweenness centrality

# Create a new graph
H = nx.Graph()

# Add the central node and its three adjacent nodes
H.add_node("A", color='deepskyblue')  # Central node
H.add_node("B", color='orchid')   # Node with degree 1
H.add_node("C", color='coral') # One of the nodes in the densely connected cluster
H.add_node("D", color='coral') # Another node in the densely connected cluster
H.add_node("F", color='lavender')# Additional node to connect with B

# Add edges for Node A
H.add_edge("A", "B")
H.add_edge("A", "C")
H.add_edge("A", "D")

# Connect Nodes C and D
H.add_edge("C", "D")

# Create a densely connected cluster around Nodes C and D
cluster_nodes = ["C", "D"]
for i in range(1, 6):  # Additional nodes in the cluster
    new_node = f"E{i}"
    H.add_node(new_node, color='navajowhite')
    cluster_nodes.append(new_node)

# Connect each node in the cluster to a couple of others
for i, node in enumerate(cluster_nodes):
    if i < len(cluster_nodes) - 1:
        H.add_edge(node, cluster_nodes[i + 1])
    if i < len(cluster_nodes) - 2:
        H.add_edge(node, cluster_nodes[i + 2])

# Connect Node B to Node F (to increase B's degree to 2)
H.add_edge("B", "F")

# Connect each node in the cluster with each other
for i in range(len(cluster_nodes)):
    for j in range(i + 1, len(cluster_nodes)):
        H.add_edge(cluster_nodes[i], cluster_nodes[j])

# Get the color for each node
colors = [H.nodes[n]['color'] for n in H.nodes]

# Draw the graph
pos = nx.spring_layout(H)  # positions for all nodes
nx.draw(H, pos, node_color=colors, with_labels=True, node_size=600)
plt.show()
