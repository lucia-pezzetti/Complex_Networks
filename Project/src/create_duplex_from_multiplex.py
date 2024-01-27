import sys
import os
"""
Takes a multiplex network and creates a duplex network from it selecting only the layers specified by the user.
"""
if len(sys.argv) < 4:
    print("Usage: %s <multiplex_edges_path> <layerID> <layerID> <output_file_path>" % sys.argv[0])
    sys.exit(1)

def select_layer_by_multiplex(file_path, LayerID):
    """
    Reads the file and selects only the second and third elements of each line
    The file format is LayerID, NodeID, NodeID, Weight
    where the first element is equal to the specified number 'LayerID'.
    """
    edges = []

    with open(file_path, 'r') as file:
        for line in file:
            layer, node1, node2, _ = line.split(sep=' ')
            # print(layer, node1, node2)
            if layer == LayerID:
                # Selecting only the second and third elements
                edges.append((node1, node2))
        
    return edges

layers_to_select = [sys.argv[2], sys.argv[3]]

for i in range(1,3):
    # Example usage: select_elements_by_first with i = 1 (you can change this value as needed)
    file_path = sys.argv[1]
    edges = select_layer_by_multiplex(file_path, layers_to_select[i-1])

    # Extract the file path from the command line argument
    output_file_path = sys.argv[4] + '_L_' + sys.argv[2] + '_' + sys.argv[3] + f'/layer{i}.txt'

    # Create the directory if it doesn't exist
    dir_name = os.path.dirname(output_file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(output_file_path, 'a') as file:
        for edge in edges:
            # Writing the second and third elements separated by a space
            file.write(f"{edge[0]} {edge[1]}\n")
