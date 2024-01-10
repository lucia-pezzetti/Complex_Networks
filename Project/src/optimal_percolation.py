#Import libraries
from percolation_basefunctions import *

def targeted_attack_adaptive(multiplex_structure, N, ranking_targeted):
    ranking_nodes = ranking_targeted(multiplex_structure, N)
    ranking_nodes_rank=sorted(ranking_nodes.items(),key= lambda x: x[1],reverse=True)
    ranking_nodes_rank_np=np.array(ranking_nodes_rank)
    ## This is the max element in the ranking
    max_element=ranking_nodes_rank_np[0][1]
    ## Take all the nodes that have the maximum rank
    top_ranking_nodes=ranking_nodes_rank_np[ranking_nodes_rank_np[:,1] == max_element]
    ## Remove one of these nodes at random
    target_ID = random.choice(top_ranking_nodes)[0]
    multiplex_prox = cPickle.loads(cPickle.dumps(multiplex_structure, -1))
    LMCGC,_=compute_LMCGC_block(multiplex_prox,N,[])
    print("Targeted strategy - using the function:",ranking_targeted.__name__)
    print("# of nodes removed, size of LMCGC")
    print(0,LMCGC)
    size_nodes = 0
    targeted_steps_attack = []
    total_target_nodes = []
    while LMCGC > np.sqrt(N):
        #Remove nodes one by one based on the targeted attack
        target_nodes = [target_ID]
        total_target_nodes.append([target_ID])
        LMCGC, multiplex_prox = compute_LMCGC_block(multiplex_prox, N, target_nodes)
        # print(size_nodes,LMCGC)
        size_nodes += 1
        targeted_steps_attack.append([size_nodes,LMCGC])
        if multiplex_prox == 0:
            break
        ranking_nodes = ranking_targeted(multiplex_prox,N)
        ranking_nodes_rank = sorted(ranking_nodes.items(),key= lambda x: x[1],reverse=True)
        ranking_nodes_rank_np = np.array(ranking_nodes_rank)
        ## This is the max element in the ranking
        max_element = ranking_nodes_rank_np[0][1]
        # print(f'Max element: {max_element}')
        ## Take all the nodes that have the maximum rank
        top_ranking_nodes = ranking_nodes_rank_np[ranking_nodes_rank_np[:,1] == max_element]
        ## Remove one of these nodes at random
        target_ID = random.choice(top_ranking_nodes)[0]

    print(size_nodes,LMCGC)

    return(total_target_nodes, size_nodes, targeted_steps_attack)