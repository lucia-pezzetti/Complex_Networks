#Import libraries
import numpy as np
import pandas as pd
import random
import seaborn
import shutil
from functools import reduce
import matplotlib.pyplot as plt
import networkx as nx
import collections
import itertools
from collections import Counter
from operator import itemgetter
import matplotlib.pyplot as plt
import os
import sys
import copy
import time
import subprocess
import heapq
import _pickle as cPickle
from  subprocess import Popen, PIPE,run


# Loading multiplex network from file (e.g. list of layers) into a dict of dictionary
def load_dataset_into_dict_of_dict(file,N):
    """
    :param file: name of the file  containing the list of layers of the multiplex
    :return: multiplex_structure: dictionary of dictionary containing the layers in the format:
    {layer_ID: {key:node; values: first_neighbours}}
    """
    multiplex_structure = dict()
    #take the path of the file
    path = os.path.split(file)[0]
    #open the file and takae all the names of the layers that are in the file
    multiplex_list= pd.read_csv(file, sep= ' ',names=['layer_name'])
    #Iterating over the layers name
    for row in multiplex_list.itertuples():
        layer_to_analyse=path+'/'+row.layer_name    
        # Loading the layer and save it in the dictionary of networks
        # e.g. {layersID:{key:node; values:[first_neighbours}}
        multiplex_structure=load_layers(multiplex_structure,layer_to_analyse,row.Index,N)
    return(multiplex_structure)


##Loading the singular networks as dictionary => {key:node; values:[first_neighbours]}
##set undir = False - > the network is directed
def load_layers(multiplex_structure,layer,index_layer,N,undir = True):

    #Loading the layer (from the edge list representation) and put it 
    layer_struct=np.loadtxt(layer, dtype=int,ndmin=2)
    multiplex_structure[index_layer]={}
    for edges in layer_struct:
        node_i = int(edges[0])
        node_j = int(edges[1])
        #If the node i exists in the multiplex, add node j as node_i neighbour
        if node_i in multiplex_structure[index_layer]:
            multiplex_structure[index_layer][node_i].append(node_j)
        else:
            multiplex_structure[index_layer][node_i]=[node_j]
        if undir == True:
            if node_j in multiplex_structure[index_layer]:
                multiplex_structure[index_layer][node_j].append(node_i)
            else:
                multiplex_structure[index_layer][node_j]=[node_i]
    #Put empty list for isolated nodes
    for i in range(0,N):
        if i not in multiplex_structure[index_layer]:
            multiplex_structure[index_layer][i]=[]
    return(multiplex_structure)

#Function that creates the networkx structure from the edge list dictionary. It returns the graph G as output
def create_graph_fromedgelist(graph_structure):
    """
    :param file: dictionary of neighbours  {key: node; values: first_neighbours}
    :return: Graph as a networkx formatt 
    """
    G=nx.Graph()
    for key in graph_structure.keys():
        for l in graph_structure[key]:
            G.add_edge(key,l)
    return(G)


def compute_connected_comp(multiplex_structure,N):
    """
    Compute the connected component of the multiplex network using the networkx library
    :param file: dictionary of dictionaries containing the multiplex networks, and the total number of nodes of the network N 
    :return: dictionary of the list of connected components on each layer, format: {key: layerID; values: list conn. components} 
    """
    nodes=[]
    [nodes.append(list(multiplex_structure[i].keys())) for i in multiplex_structure]
    unique_nodes=list(set([item for sublist in nodes for item in sublist]))
    list_conn_components= dict()
    conn_components=dict()
    total_nodes=[i for i in range(0,N)]
    for layer_ID in multiplex_structure:
        list_conn_components[layer_ID] = dict.fromkeys(total_nodes,-100000)
        G = create_graph_fromedgelist(multiplex_structure[layer_ID])
        ##Sort the connected components of the graph  by size, and assign an ID to each node according to the size of cluster
        ##Components that are isolated have ID equal to -100000
        ##The component ID starts from 1 
        conn_components[layer_ID]=sorted(nx.connected_components(G), key = len, reverse=True)
        s= len(conn_components[layer_ID])
        for number_component in range(1,s+1):
            for l in conn_components[layer_ID][number_component-1]:
                list_conn_components[layer_ID][l]=number_component
    return(list_conn_components)


#Compute the connected component of a specific "layer_ID" layer and update the dictionary list_conn_components
def compute_connected_comp_layer(multiplex_structure,N,layer_ID,list_conn_components):
    """
    Compute the connected component of a specific "layer_ID" layer and update the dictionary list_conn_components
    :param file: dictionary of dictionaries containing the multiplex networks, the number of nodes N, layer_ID for computing the conn.
    component,  dictionary of the list of connected components on each layer, format: {key: layerID; values: list conn. components} 
    :return: updated dictionary of the list of connected components on each layer, format: {key: layerID; values: list conn. components} 
    """
    nodes=[]
    M=len(multiplex_structure)
    [nodes.append(list(multiplex_structure[i].keys())) for i in multiplex_structure]
    unique_nodes=list(set([item for sublist in nodes for item in sublist]))
    conn_components=dict()
    total_nodes=[i for i in range(0,N)]
    G = create_graph_fromedgelist(multiplex_structure[layer_ID])
    #print("computing components")
    conn_components[layer_ID]=sorted(nx.connected_components(G), key = len, reverse=True)
    s= len(conn_components[layer_ID])
    for number_component in range(1,s+1):
        for l in conn_components[layer_ID][number_component-1]:
            list_conn_components[layer_ID][l]=number_component
    return(list_conn_components)

#Compute the product of the degrees of each layer and return it as a dictionary nodeID:Product degrees 
def computing_product_degree(multiplex_structure,N):
    """
    Compute the product of the degrees of each layer and return it as a dictionary nodeID:Product degrees 
    :param file: dictionary of dictionaries containing the multiplex networks, the number of nodes N
    :return: dictionary containing the product of the degrees of each layers, format: nodeID:Product degrees 
    """
    total_nodes=[i for i in range(0,N)]
    total_degree = dict.fromkeys(total_nodes, 0)
    for layer_ID in multiplex_structure:
        for nodes in multiplex_structure[layer_ID]:
            if total_degree[nodes] == 0:
                total_degree[nodes] = len(multiplex_structure[layer_ID][nodes])
            else:
                total_degree[nodes]*=len(multiplex_structure[layer_ID][nodes])
    return(total_degree)


#### Slow implementation for computing the LMCGC of a multiplex networks ####
#### For duplex up to ~ 10^4 nodes is still fine                         #### 
#Compute the largest cluster of the Mutually Connected Giant Component
#Multiplex structure: dict of dict
#N: number of nodes
#Block_target: list of nodes to be removed from the multiplex
def compute_LMCGC_block(multiplex_structure, N, block_target):
    """
    Compute the largest cluster of the Mutually Connected Giant Component
    :param file: dictionary of dictionaries containing the multiplex networks, the number of nodes N, the list of nodes to be removed from the multiplex
    :return: the size of the largest cluster of the Mutually Connected Giant Component, the percolated multiplex 
    """
    multiplex_new = cPickle.loads(cPickle.dumps(multiplex_structure, -1))
    #Removing the connections of the target in all the layers:
    #print(block_target)
    for target in block_target:
        for layer_ID in multiplex_structure:
            # check if the target is in the adjacency list of the layer
            if target in multiplex_structure[layer_ID]:
                # iterate over the neighbours of the target
                for j in multiplex_structure[layer_ID][target]:
                    if j in multiplex_new[layer_ID][target]:
                        # remove the neighbour from the adjacency list of the target
                        multiplex_new[layer_ID][target].remove(j)
                    if target in multiplex_new[layer_ID][j]:
                        # remove the target from the adjacency list of the neighbour
                        multiplex_new[layer_ID][j].remove(target)
    multiplex_next = cPickle.loads(cPickle.dumps(multiplex_new, -1))

    #Now computing the MCGC in the multiplex network [using the algorithm of V.Buldyrev 2010 - Nature -> Not optimised]
    while(1):
        flag = 0
        #multiplex_new = copy.deepcopy(multiplex_next)
        multiplex_new = cPickle.loads(cPickle.dumps(multiplex_next, -1))
        #Computing the connected components of all the layers in the multiplex
        list_connected_components = compute_connected_comp(multiplex_new, N)                   
        for layer_ID in multiplex_new:
            for i in multiplex_new[layer_ID].keys():
                for j in multiplex_new[layer_ID][i]:
                    if i < j:
                        for layer_check in range(0, len(multiplex_new)):
                            #remove the links that don't belong to the same component in the different layers
                            if list_connected_components[layer_check][i] != list_connected_components[layer_check][j] or list_connected_components[layer_check][i] == -100000 or list_connected_components[layer_check][j] == -100000:
                                    multiplex_next[layer_ID][i].remove(j)
                                    multiplex_next[layer_ID][j].remove(i)
                                    flag = 1
            #Computing the connected components of all the layers in the multiplex
            list_connected_components = compute_connected_comp_layer(multiplex_new,N,layer_ID,list_connected_components)
        #If no changes happened in a cycle, then exit!
        if flag == 0:
            break
    G = create_graph_fromedgelist(multiplex_next[0])
    #Check the size of the connected component of one of the layers, after all the link removals, the giant component represents the MCC
    conn_components = sorted(nx.connected_components(G), key = len, reverse=True)
    if not conn_components:
        return(0, 0)
    else:
        return(len(conn_components[0]), multiplex_next)

        

# Function computing the total degree of each node of the multiplex structure, in addition
# it also computes the  participation coefficient [See Nicosia-Battiston-Latora 2014 - Structural measures
# for multiplex networks for all the details]
def computing_total_degree_and_participation(multiplex_structure, N):
    """
    :param multiplex_structure: structure containing the multiplex (dictionary of dictionary) 
        as from the function load_dataset_into_dict_of_dict
            N : number of total nodes in the multiplex
    :return:
            total_degree: dictionary {node: total_degree}
            participation_degree: dictionary {node: participation coefficient (of the degree)}
            dgr_part: dictionary: {node: [total_degree[node],participation_coefficient[node]]}
    """
    nodes = []

    #multiplex_structure_aux=copy.deepcopy(multiplex_structure)
    multiplex_structure_aux = cPickle.loads(cPickle.dumps(multiplex_structure, -1))
    M = len(multiplex_structure)

    [nodes.append(list(multiplex_structure_aux[i].keys())) for i in multiplex_structure_aux]

    #Extract the nodes that are active in the multiplex network
    unique_nodes = list(set([item for sublist in nodes for item in sublist]))
    total_nodes = [i for i in range(0,N)]
    total_degree = dict.fromkeys(total_nodes, 0)

    degree_layers_multiplex = {}
    for layer_ID in multiplex_structure_aux:
        degree_layers_multiplex[layer_ID] = dict.fromkeys(total_nodes,0)
        for nodes in multiplex_structure_aux[layer_ID]:
            total_degree[nodes] += len(multiplex_structure_aux[layer_ID][nodes])
    #Total degree is a dictionary containing the sum of the degree of each nodes in all the layers
                
    participation_degree = dict.fromkeys(total_nodes, 0)
    for node in unique_nodes:
        temp_sum = 0.0
        for layer_ID in multiplex_structure_aux:
            if node in multiplex_structure_aux[layer_ID] and total_degree[node] != 0:
                degree_node = len(multiplex_structure_aux[layer_ID][node])
                degree_layers_multiplex[layer_ID][node] = degree_node
                temp_sum += ((1.0*degree_node) / (1.0*total_degree[node]))**2
        participation_degree[node] = (M/(M-1.0)) * (1-temp_sum)
        if node in multiplex_structure_aux[layer_ID] and total_degree[node] == 0:
            participation_degree[node] = 0

    dgr_part = dict({key: [total_degree[key], participation_degree[key]] for key in participation_degree})
    return(total_degree, participation_degree, dgr_part, degree_layers_multiplex)


#Compute the intersection graph of the multiplex structure
def find_intersection_graph(multiplex_structure, N):
    intersection_graph={}
    for i in range(N):
        if i in multiplex_structure[0]:
            all_neighbour = multiplex_structure[0][i]
        else:
            continue
        #find all the neighbours of node i in all the layers
        for k in multiplex_structure:
            all_neighbour = set(all_neighbour).intersection(multiplex_structure[k][i])
        intersection_graph[i] = list(all_neighbour)
    return(intersection_graph)

#Compute the union graph of the multiplex structure
def find_union_graph(multiplex_structure, N):
    union_graph={}
    for i in range(N):
        if i in multiplex_structure[0]:
            if multiplex_structure[0][i]!= []:
                all_neighbour=multiplex_structure[0][i]
            else:
                all_neighbour=[]
        else:
            continue
        #find all the neighbours of node i in all the layers
        for k in multiplex_structure:
            all_neighbour=set(all_neighbour).union(multiplex_structure[k][i])
        union_graph[i]=list(all_neighbour)
    return(union_graph)


#Compute the disjunt union graph of the multiplex structure
def find_symmetric_difference_graph(multiplex_structure,N):
    symmetric_graph={}
    for i in range(N):
        if i in multiplex_structure[0]:
            all_neighbour=multiplex_structure[0][i]
        else:
            continue
        for k in multiplex_structure:
            all_neighbour=set(all_neighbour).symmetric_difference(multiplex_structure[k][i])
        symmetric_graph[i]=list(all_neighbour)
    return(symmetric_graph)


# Compute the EMD 1step ranking as in Eq. 4 of the paper "Targeted damage to interdependent networks" 
# by G. J. Baxter, G. Timar, and J. F. F. Mendes, PRE 98, 032307 (2018) 
def EMD1step_ranking(multiplex_structure, N):
    total_nodes=[i for i in range(0,N)]
    emd_ranking=dict.fromkeys(total_nodes,0)
    total_degree,participation_degree,dgr_part,degree_layers_multiplex = computing_total_degree_and_participation(multiplex_structure,N)
    for layerID in multiplex_structure:
        for nodei in multiplex_structure[layerID]:
            temp = 0
            for neigh_of_i in multiplex_structure[layerID][nodei]:
                if degree_layers_multiplex[layerID][neigh_of_i] !=0:
                    activity = np.heaviside(degree_layers_multiplex[layerID][neigh_of_i],0)+np.heaviside(degree_layers_multiplex[(layerID+1)%2][neigh_of_i],0)
                    temp+=((1./activity)*(total_degree[neigh_of_i]/degree_layers_multiplex[layerID][neigh_of_i]))
            emd_ranking[nodei]+=temp
    return(emd_ranking)




# Compute the EMD ranking as in Eq. 1 of the paper "Targeted damage to interdependent networks" 
# by G. J. Baxter, G. Timar, and J. F. F. Mendes, PRE 98, 032307 (2018) 
# The criterion for the solutions to have fully converged is that the largest
# relative difference in all values is less than 10e-7
def EMDfullsteps_ranking(multiplex_structure, N):
    nodes=[]
    M=len(multiplex_structure)
    [nodes.append(list(multiplex_structure[i].keys())) for i in multiplex_structure]
    unique_nodes=list(set([item for sublist in nodes for item in sublist]))
    total_nodes=[i for i in range(0,N)]
    emd_ranking=dict.fromkeys(total_nodes,0)
    total_degree,participation_degree,dgr_part,degree_layers_multiplex = computing_total_degree_and_participation(multiplex_structure,N)
    for layerID in multiplex_structure:
        for nodei in multiplex_structure[layerID]:
            temp = 0
            for neigh_of_i in multiplex_structure[layerID][nodei]:
                activity = np.heaviside(degree_layers_multiplex[layerID][neigh_of_i],0)+np.heaviside(degree_layers_multiplex[(layerID+1)%2][neigh_of_i],0)
                temp+=((1./activity)*(total_degree[neigh_of_i]/degree_layers_multiplex[layerID][neigh_of_i]))
            emd_ranking[nodei]+=temp
    while(1):
        emd_ranking_new=dict.fromkeys(total_nodes,0)
        for layerID in multiplex_structure:
            for nodei in multiplex_structure[layerID]:
                temp = 0
                for neigh_of_i in multiplex_structure[layerID][nodei]:
                    activity = np.heaviside(degree_layers_multiplex[layerID][neigh_of_i],0)+np.heaviside(degree_layers_multiplex[(layerID+1)%2][neigh_of_i],0)
                    temp+=((1./activity)*(emd_ranking[neigh_of_i]/degree_layers_multiplex[layerID][neigh_of_i]))
                emd_ranking_new[nodei]+=temp
        ranking=np.array(list(emd_ranking.values()))
        ranking_new=np.array(list(emd_ranking_new.values()))
        #The elements are ordered in the array 
        diff_toll=np.abs(ranking_new-ranking)
        max_difference=max([diff_toll[i]/ranking[i] for i in range(len(diff_toll)) if ranking[i] !=0])
        if  max_difference < 1e-7:
            emd_ranking=cPickle.loads(cPickle.dumps(emd_ranking_new, -1))
            #emd_ranking =copy.deepcopy(emd_ranking_new)
            break
        #emd_ranking =copy.deepcopy(emd_ranking_new)
        emd_ranking =cPickle.loads(cPickle.dumps(emd_ranking_new, -1))
    return(emd_ranking)


### Compute the non-normalized betwenneess centrality of a specific layer "layerID" in the multiplex
def compute_betweenness_centrality(multiplex_structure, layerID, N):
    G = create_graph_fromedgelist(multiplex_structure[layerID])
    list_centrality = nx.betweenness_centrality(G, normalized=False)
    for i in range(0, N):
        if i not in list_centrality:
            list_centrality[i] = 0
    return(list_centrality)


### Compute the non-rescaled betwenneess centrality of a single-layer graph
def compute_betweenness_centrality_single_layer_network(graph_structure, N):
    # construct the graph from the multiplex structure
    G = create_graph_fromedgelist(graph_structure)
    # evaluate the betweenness centrality
    list_centrality = nx.betweenness_centrality(G, normalized=False)

    # handle the case of isolated nodes
    for i in range(0, N):
        if i not in list_centrality:
            list_centrality[i] = 0

    return(list_centrality)


### Compute the non-normalized Katz centrality of a specific layer "layerID" in the multiplex
def compute_katz_centrality(multiplex_structure, layerID, N, alpha):
    G = create_graph_fromedgelist(multiplex_structure[layerID])
    list_centrality = nx.katz_centrality(G, alpha=alpha, normalized=False)

    for i in range(0, N):
        if i not in list_centrality:
            list_centrality[i] = 0
    return(list_centrality)

### Compute the non-normalized Katz centrality of a single-layer graph
def compute_katz_centrality_single_layer_network(graph_structure, N, alpha):
    # construct the graph from the multiplex structure
    G = create_graph_fromedgelist(graph_structure)
    # evaluate the Katz centrality
    list_centrality = nx.katz_centrality(G, alpha=alpha, normalized=False)

    # handle the case of isolated nodes
    for i in range(0, N):
        if i not in list_centrality:
            list_centrality[i] = 0

    return(list_centrality)

### Compute the harmonic centrality of a specific layer "layerID" in the multiplex
### Harmonic centrality is defined as the sum of the reciprocal of the shortest path distances from all other nodes to a given node
### It is a modification of the closeness centrality that naturally handle the case of disconnected graphs
def compute_harmonic_centrality(multiplex_structure, layerID, N):
    G = create_graph_fromedgelist(multiplex_structure[layerID])
    list_centrality = nx.harmonic_centrality(G)

    for i in range(0, N):
        if i not in list_centrality:
            list_centrality[i] = 0
    return(list_centrality)

### Compute the non-normalized harmonic centrality of a single-layer graph
def compute_harmonic_centrality_single_layer_network(graph_structure, N):
    # construct the graph from the multiplex structure
    G = create_graph_fromedgelist(graph_structure)
    # evaluate the Katz centrality
    list_centrality = nx.harmonic_centrality(G)

    # handle the case of isolated nodes
    for i in range(0, N):
        if i not in list_centrality:
            list_centrality[i] = 0

    return(list_centrality)

def get_alpha(multiplex_structure, N):
    G1 = create_graph_fromedgelist(multiplex_structure[0])
    G2 = create_graph_fromedgelist(multiplex_structure[1])
    aggregate_network=create_graph_fromedgelist(find_union_graph(multiplex_structure,N))
    intersection_graph=create_graph_fromedgelist(find_intersection_graph(multiplex_structure,N))

    # Get the adjacency matrix
    A1 = nx.adjacency_matrix(G1)
    A2 = nx.adjacency_matrix(G2)
    A_aggregate=nx.adjacency_matrix(aggregate_network)
    A_intersection=nx.adjacency_matrix(intersection_graph)

    # Calculate the eigenvalues
    eigenvalues1 = np.linalg.eigvals(A1.toarray())
    eigenvalues2 = np.linalg.eigvals(A2.toarray())
    eigenvalues_aggregate=np.linalg.eigvals(A_aggregate.toarray())
    eigenvalues_intersection=np.linalg.eigvals(A_intersection.toarray())

    # Find the largest eigenvalue
    lambda_max1 = max(eigenvalues1.real)
    lambda_max2 = max(eigenvalues2.real)
    lambda_max_aggregate=max(eigenvalues_aggregate.real)
    lambda_max_intersection=max(eigenvalues_intersection.real)

    # Calculate the alpha
    alpha = min(1/lambda_max1, 1/lambda_max2, 1/lambda_max_aggregate, 1/lambda_max_intersection) - 0.01
    return(alpha)


## Compute the Euclidean distance between two points given in input
def euclidean_distance(a,b):
    n = len(a)
    distance = 0
    for i in range(n):
        distance += (a[i]-b[i])**2
    return (np.sqrt(distance))



#### Ranking based on the generalisation of CI in the case of duplex network:
###\sum_\alpha \sum_{j in N_i^\alpha} \frac{  k_j^\alpha*k_j^\beta - k_j^{intersect} }{  k_j^{\alpha} }
def compute_ranking_CI_duplex(multiplex_structure,N):
    total_nodes=[i for i in range(N)]
    fitness=dict.fromkeys(total_nodes,0)
    total_degree,participation_degree,dgr_part,degree_layers_multiplex = computing_total_degree_and_participation(multiplex_structure,N)
    aggregate_network=find_union_graph(multiplex_structure,N)
    intersection_graph=find_intersection_graph(multiplex_structure,N)
    ranking={}
    for layerID in multiplex_structure:
        for nodei in multiplex_structure[layerID]:
            temp=0
            for neigh_of_i in multiplex_structure[layerID][nodei]:
                if degree_layers_multiplex[layerID][neigh_of_i] != 0:
                    numerator1=(degree_layers_multiplex[(layerID+1)%2][neigh_of_i])*(degree_layers_multiplex[layerID][neigh_of_i])
                    numerator2=len(intersection_graph[neigh_of_i])
                    numerator=numerator1-numerator2
                    denominator=degree_layers_multiplex[layerID][neigh_of_i]
                    temp+= (numerator/denominator)
            fitness[nodei]+=temp
    for nodei in fitness:
        if len(aggregate_network[nodei])!=0:
            numerator1=(degree_layers_multiplex[0][nodei])*(degree_layers_multiplex[1][nodei])
            numerator2=len(intersection_graph[nodei])
            numerator=numerator1-numerator2
            denominator=len(aggregate_network[nodei])
            fitness[nodei]=fitness[nodei] * (numerator/denominator)
    return(fitness)

#### Ranking based on the generalisation of CI in the case of duplex network
def compute_ranking_DCI(multiplex_structure, N):
    total_nodes = [i for i in range(N)]
    fitness = dict.fromkeys(total_nodes, 0)
    total_degree, participation_degree, dgr_part, degree_layers_multiplex = computing_total_degree_and_participation(multiplex_structure, N)
    
    aggregate_network = find_union_graph(multiplex_structure, N)
    intersection_graph = find_intersection_graph(multiplex_structure, N)
    
    # second term of the product of the generalised DCI
    for layerID in multiplex_structure:
        for nodei in multiplex_structure[layerID]:
            for neigh_of_i in multiplex_structure[layerID][nodei]:
                if degree_layers_multiplex[layerID][neigh_of_i] != 0:
                    # for every node adjacent to nodei in the current layer, we add the degree of the node in the other layer - 1
                    fitness[nodei] += degree_layers_multiplex[(layerID+1)%2][neigh_of_i] - 1 
    
    for nodei in fitness:
        if len(aggregate_network[nodei]) != 0:
            # first term of the product of the generalised DCI
            numerator1 = (degree_layers_multiplex[0][nodei]) * (degree_layers_multiplex[1][nodei])
            numerator2 = len(intersection_graph[nodei])
            numerator = numerator1 - numerator2
            denominator = len(aggregate_network[nodei])

            # multiply the two terms together
            fitness[nodei] = fitness[nodei] * (numerator/denominator)
    return(fitness)


#### Ranking based on the generalisation of CI with Katz centrality in the case of duplex network
def compute_ranking_DCI_katz_centrality(multiplex_structure, N, alpha=0.7):
    total_nodes = [i for i in range(N)]
    fitness = dict.fromkeys(total_nodes, 0)
    total_degree, participation_degree, dgr_part, degree_layers_multiplex = computing_total_degree_and_participation(multiplex_structure, N)
    
    # compute the katz centrality of each node in each layer
    alpha = get_alpha(multiplex_structure, N)
    katz_centrality_multiplex = dict.fromkeys(total_nodes, 0)
    katz_centrality_multiplex[0] = compute_katz_centrality(multiplex_structure, 0, N, alpha)
    katz_centrality_multiplex[1] = compute_katz_centrality(multiplex_structure, 1, N, alpha)

    aggregate_network = find_union_graph(multiplex_structure, N)
    katz_centrality_aggregate = compute_katz_centrality_single_layer_network(aggregate_network, N, alpha)
    intersection_graph = find_intersection_graph(multiplex_structure, N)
    katz_centrality_intersection = compute_katz_centrality_single_layer_network(intersection_graph, N, alpha)
    
    # second term of the product of the generalised DCI
    for layerID in multiplex_structure:
        for nodei in multiplex_structure[layerID]:
            for neigh_of_i in multiplex_structure[layerID][nodei]:
                if degree_layers_multiplex[layerID][neigh_of_i] != 0:
                    # for every node adjacent to nodei in the current layer, we add the katz centrality of the node in the other layer - 1
                    fitness[nodei] += katz_centrality_multiplex[(layerID+1)%2][neigh_of_i] - 1

    # for nodei in fitness:
        if katz_centrality_aggregate[nodei] != 0:
            # first term of the product of the generalised DCI
            numerator1 = (katz_centrality_multiplex[0][nodei]) * (katz_centrality_multiplex[1][nodei])
            numerator2 = katz_centrality_intersection[nodei] 
            numerator = numerator1 - numerator2
            denominator = katz_centrality_aggregate[nodei]

            # multiply the two terms together
            fitness[nodei] = fitness[nodei] * (numerator/denominator)

    return(fitness)

#### Ranking based on the generalisation of CI with harmonic centrality in the case of duplex network
def compute_ranking_DCI_harmonic_centrality(multiplex_structure, N):
    total_nodes = [i for i in range(N)]
    fitness = dict.fromkeys(total_nodes, 0)
    total_degree, participation_degree, dgr_part, degree_layers_multiplex = computing_total_degree_and_participation(multiplex_structure, N)


    # compute the eigenvector centrality of each node in each layer
    harmonic_centrality_multiplex = dict.fromkeys(total_nodes, 0)
    harmonic_centrality_multiplex[0] = compute_harmonic_centrality(multiplex_structure, 0, N)
    harmonic_centrality_multiplex[1] = compute_harmonic_centrality(multiplex_structure, 1, N)

    aggregate_network = find_union_graph(multiplex_structure, N)
    harmonic_centrality_aggregate = compute_harmonic_centrality_single_layer_network(aggregate_network, N)
    intersection_graph = find_intersection_graph(multiplex_structure, N)
    harmonic_centrality_intersection = compute_harmonic_centrality_single_layer_network(intersection_graph, N)
    
    # second term of the product of the generalised DCI
    for layerID in multiplex_structure:
        for nodei in multiplex_structure[layerID]:
            for neigh_of_i in multiplex_structure[layerID][nodei]:
                if degree_layers_multiplex[layerID][neigh_of_i] != 0:
                    # for every node adjacent to nodei in the current layer, we add the eigenvector centrality of the node in the other layer - 1
                    fitness[nodei] += harmonic_centrality_multiplex[(layerID+1)%2][neigh_of_i]
    
    for nodei in fitness:
        if harmonic_centrality_aggregate[nodei] != 0:
            # first term of the product of the generalised DCI
            numerator1 = (harmonic_centrality_multiplex[0][nodei] + 1) * (harmonic_centrality_multiplex[1][nodei] + 1)
            numerator2 = harmonic_centrality_intersection[nodei] + 1
            numerator = numerator1 - numerator2
            denominator = harmonic_centrality_aggregate[nodei] + 1

            # multiply the two terms together
            fitness[nodei] = fitness[nodei] * (numerator/denominator)

    return(fitness)


#### Ranking based on the generalisation of CI in the case of duplex network (with the modification for isolated nodes)
def compute_ranking_DCIz(multiplex_structure,N):
    total_nodes = [i for i in range(N)]
    fitness = dict.fromkeys(total_nodes,0)
    total_degree,participation_degree,dgr_part,degree_layers_multiplex = computing_total_degree_and_participation(multiplex_structure,N)
    aggregate_network=find_union_graph(multiplex_structure,N)
    intersection_graph=find_intersection_graph(multiplex_structure,N)
    # ranking={}
    for layerID in multiplex_structure:
        for nodei in multiplex_structure[layerID]:
            # temp=0
            for neigh_of_i in multiplex_structure[layerID][nodei]:
                if degree_layers_multiplex[layerID][neigh_of_i] != 0:
                    fitness[nodei] += degree_layers_multiplex[(layerID+1)%2][neigh_of_i] - 1
                    # fitness[nodei]+=temp
    for nodei in fitness:
        if len(aggregate_network[nodei])!=0:
            numerator1 = (degree_layers_multiplex[0][nodei]+1) * (degree_layers_multiplex[1][nodei]+1)
            numerator2 = 3 * len(intersection_graph[nodei])
            numerator = numerator1 - numerator2 - 1
            denominator = len(aggregate_network[nodei])
            fitness[nodei] = fitness[nodei] * (numerator/denominator)
    return(fitness)


#### Ranking based on the generalisation of CI with betweenness centrality in the case of duplex network
def compute_ranking_DCIz_betweenness_centrality(multiplex_structure, N):
    total_nodes = [i for i in range(N)]
    fitness = dict.fromkeys(total_nodes, 0)
    total_degree, participation_degree, dgr_part, degree_layers_multiplex = computing_total_degree_and_participation(multiplex_structure, N)


    # compute the betweenness centrality of each node in each layer
    betweenness_centrality_multiplex = dict.fromkeys(total_nodes, 0)
    betweenness_centrality_multiplex[0] = compute_betweenness_centrality(multiplex_structure, 0, N)
    betweenness_centrality_multiplex[1] = compute_betweenness_centrality(multiplex_structure, 1, N)

    aggregate_network = find_union_graph(multiplex_structure, N)
    betweenness_centrality_aggregate = compute_betweenness_centrality_single_layer_network(aggregate_network, N)
    intersection_graph = find_intersection_graph(multiplex_structure, N)
    betweenness_centrality_intersection = compute_betweenness_centrality_single_layer_network(intersection_graph, N)
    
    # # second term of the product of the generalised DCI
    for layerID in multiplex_structure:
        for nodei in multiplex_structure[layerID]:
            for neigh_of_i in multiplex_structure[layerID][nodei]:
                if degree_layers_multiplex[layerID][neigh_of_i] != 0:
                    # for every node adjacent to nodei in the current layer, we add the betweenness centrality of the node in the other layer - 1
                    fitness[nodei] += ((2 * betweenness_centrality_multiplex[(layerID+1)%2][neigh_of_i] + 2)/((N-1)*(N-2)))
    
    for nodei in fitness:
        if betweenness_centrality_aggregate[nodei] != 0:
            # first term of the product of the generalised DCI
            numerator1 = (betweenness_centrality_multiplex[0][nodei] + 2) * (betweenness_centrality_multiplex[1][nodei] + 2)
            numerator2 = 3 * (betweenness_centrality_intersection[nodei] + 1)
            numerator = 2 * (numerator1 - numerator2 - 1)/((N-1)*(N-2))
            denominator = 2 * (betweenness_centrality_aggregate[nodei] + 1)/((N-1)*(N-2))

            # multiply the two terms together
            fitness[nodei] = fitness[nodei] * (numerator/denominator)
    
    return(fitness)
    

def compute_ranking_product_katz_centrality(multiplex_structure, N):
    """
    Compute for each node the product of its katz centralities in the different layers
    :param file: dictionary of dictionaries containing the multiplex networks, the number of nodes N
    :return: dictionary containing the product of the Katz centralities of each node, format: nodeID:prod_katz_centrality 
    """
    total_nodes=[i for i in range(0,N)]
    total_degree = dict.fromkeys(total_nodes, 0)
    alpha = get_alpha(multiplex_structure, N)

    # compute the katz centrality of each node in each layer
    katz_centrality_multiplex = dict.fromkeys(total_nodes, 0)
    katz_centrality_multiplex[0] = compute_katz_centrality(multiplex_structure, 0, N, alpha)
    katz_centrality_multiplex[1] = compute_katz_centrality(multiplex_structure, 1, N, alpha)

    for layer_ID in multiplex_structure:
        for nodes in multiplex_structure[layer_ID]:
            if total_degree[nodes] == 0:
                total_degree[nodes] = katz_centrality_multiplex[layer_ID][nodes]
            else:
                total_degree[nodes] *= katz_centrality_multiplex[layer_ID][nodes]
    return(total_degree)


def compute_ranking_product_harmonic_centrality(multiplex_structure, N):
    """
    Compute for each node the product of its harmonic centralities in the different layers
    :param file: dictionary of dictionaries containing the multiplex networks, the number of nodes N
    :return: dictionary containing the product of the harmonic centralities of each node, format: nodeID:prod_katz_centrality 
    """
    total_nodes=[i for i in range(0,N)]
    total_degree = dict.fromkeys(total_nodes, 0)

    # compute the harmonic centrality of each node in each layer
    harmonic_centrality_multiplex = dict.fromkeys(total_nodes, 0)
    harmonic_centrality_multiplex[0] = compute_harmonic_centrality(multiplex_structure, 0, N)
    harmonic_centrality_multiplex[1] = compute_harmonic_centrality(multiplex_structure, 1, N)

    for layer_ID in multiplex_structure:
        for nodes in multiplex_structure[layer_ID]:
            if total_degree[nodes] == 0:
                total_degree[nodes] = harmonic_centrality_multiplex[layer_ID][nodes]
            else:
                total_degree[nodes] *= harmonic_centrality_multiplex[layer_ID][nodes]
    return(total_degree)


#### Slow implementation of the Collective Influence algorithm with l=2 (The original code is much faster using a
#### max-heap data structure)
#### presented in the paper: Morone, F., Makse, H. Influence maximization in complex networks
#### through optimal percolation. Nature 524, 65-68 (2015). https://doi.org/10.1038/nature14604
def compute_CI_2_ranking(multiplex_structure,N):
    total_nodes=[i for i in range(N)]
    fitness={}
    total_degree,participation_degree,dgr_part,degree_layers_multiplex = computing_total_degree_and_participation(multiplex_structure,N)
    for layerID in multiplex_structure:
        fitness[layerID] = dict.fromkeys(total_nodes,0)
        for nodei in multiplex_structure[layerID]:
            temp=0
            for neigh_of_i in multiplex_structure[layerID][nodei]:
                for second_neigh in multiplex_structure[layerID][neigh_of_i]:
                    temp += (degree_layers_multiplex[layerID][second_neigh]-1)
                       
            fitness[layerID][nodei] = temp*(degree_layers_multiplex[layerID][nodei]-1)
    return(fitness)
