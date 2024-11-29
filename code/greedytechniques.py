import networkx as nx
import networkx as nx
import random
G1 = nx.read_edgelist("/home/project/anaconda3/pycharm-community-2023.1.2/differentialprivacy_experimentation/differentialprivacy_experimentation/data/cora/train_0.5", nodetype=int)
num_nodes1 = G1.number_of_nodes()
num_edges1 = G1.number_of_edges()
print(num_nodes1,num_edges1 )


def greedy_influence_maximization(G, k):
    """
    Greedy algorithm for influence maximization on a graph G.

    Args:
        G: a networkx graph object.
        k: the number of nodes to select.

    Returns:
        A list of k nodes that maximizes the spread of influence.
    """

    S = [] # Selected nodes
    for i in range(k):
        max_node = None
        max_influence = -1
        for node in G.nodes():
            if node not in S:
                influence = get_influence(G, S + [node])
                if influence > max_influence:
                    max_node = node
                    max_influence = influence
        S.append(max_node)
    return S

def get_influence(G, S):
    """
    Compute the spread of influence on a graph G starting from a set of seed nodes S.

    Args:
        G: a networkx graph object.
        S: a list of seed nodes.

    Returns:
        The number of nodes influenced by S.
    """

    H = nx.Graph(G) # Copy the original graph
    for node in S:
        H.nodes[node]['activated'] = True # Activate the seed nodes

    activated_nodes = set(S)
    while True:
        activated_nodes_old = set(activated_nodes)
        for node in activated_nodes_old:
            for neighbor in H.neighbors(node):
                if not H.nodes[neighbor].get('activated', False):
                    # Check if the neighbor is influenced
                    num_activated_neighbors = sum(1 for n in H.neighbors(neighbor) if H.nodes[n].get('activated', False))
                    if random.random() < 1 - (1 - H[node][neighbor]['weight']) ** num_activated_neighbors:
                        H.nodes[neighbor]['activated'] = True
                        activated_nodes.add(neighbor)
        if activated_nodes == activated_nodes_old:
            break
    print(len(activated_nodes))
    return len(activated_nodes)
# Set the weights of the edges (optional)
for u, v in G1.edges():
    G1[u][v]['weight'] = 1

# Run the greedy algorithm
k =100
S1_G = greedy_influence_maximization(G1, k)

# Print the selected nodes
print(S1_G)


def independent_cascade(G, seeds, p):
    # Set the initial active nodes to the seed nodes
    active_nodes = list(seeds)

    # Set the initial round to 0
    round = 0

    # Repeat until no more nodes can become active
    while True:
        # Set the nodes that became active in this round to an empty set
        newly_active_nodes = set()

        # For each active node
        for node in active_nodes:
            # For each of its neighbors that is not already active
            for neighbor in G.neighbors(node):
                if neighbor not in active_nodes and neighbor not in newly_active_nodes:
                    # Add the neighbor to the set of newly active nodes with probability p
                    if random.random() < p:
                        newly_active_nodes.add(neighbor)

        # If no nodes became active in this round, exit the loop
        if not newly_active_nodes:
            break

        # Add the newly active nodes to the set of active nodes
        active_nodes.extend(list(newly_active_nodes))

        # Increment the round counter
        round += 1

    # Return the number of nodes that became active during the simulation
    return len(active_nodes)
seeds = S1_G
Spread1_G = independent_cascade(G1, seeds, 0.1)
print("spread",Spread1_G)
