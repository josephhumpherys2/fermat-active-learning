"""
Note:
        s2query is direct implementation by @jifanz
        https://github.com/jifanz/GALAXY

        s2query_weighted() and RunS2() are adapted from that implementation
"""


from graph import *
import numpy as np


def s2query(graph):
    dist, path = graph.shortest_shortest_path()
    if path is None:
        query = graph.not_queried[np.random.randint(len(graph.not_queried))]
    else:
        assert len(path) > 2, "Found path connecting oppositely label nodes."
        query = path[len(path) // 2]
    return query.idx


def s2query_weighted(graph, weight_dictionary):
    """
    Parameters:
        graph (graph object from graph.py):
            Object from graph.py

        weight_dictionary (dictionary): 
            Dictionary of weight**p for each valid edge(least to greatest)
        
        """

    # Run s2 algorithm 
    dist, path = graph.shortest_shortest_path(weight_dictionary=weight_dictionary, weighted=True)

    if path is None:
        # Randomly label point
        query = graph.not_queried[np.random.randint(len(graph.not_queried))]
        return query.idx

    # Find node after at least half of weight is reached
    edge_arr = []

    for i in range(len(path)-1):
        start = path[i]
        end = path[i+1]
        start_idx = start.idx
        end_idx = end.idx

        if start_idx < end_idx:
            edge_arr.append(weight_dictionary[start_idx][end_idx])
        else:
            edge_arr.append(weight_dictionary[end_idx][start_idx])
        

    sum_edge = sum(edge_arr)
    mid = sum_edge / 2

    # If the first element is greater than half of the total
    if edge_arr[0] >= mid:
        # Return the second element
        return path[1].idx
    # If the last element is greater than half of the total
    if edge_arr[-1] >= mid:
        # Return the second to last element
        return path[-2].idx

    iter_sum = 0
    for j in range(len(edge_arr)):
        iter_sum += edge_arr[j]
        if iter_sum >= mid:
            return path[j+1].idx
    

def RunS2(graph, p, initials, nx_edges, budget, weighted=False):
    """
    Parameters:
        graph (graph object from graph.py):
            The graph we are running the algorithm on

        p (float): 
            The power to take each weighted edge to
        
        initials (array): 
            The initial points you want labeled
            
        nx_edges (dictionary):
            Dictionary of edge weights from nx.graph library
            e.g. nx_edges = nx.get_edge_attributes(nx_graph, 'weight')

        weighted (bool):
            Whether the algorithm is being run on a weighted graph

        budget (int): 
            The number of iterations to run

    """

    # Set up weight dictionary
    weight_dictionary = {}
    for i in list(nx_edges.keys()):
        start = i[0]
        end = i[1]
        if start not in weight_dictionary:
            weight_dictionary[start] = {}
        weight_dictionary[start][end] = nx_edges[i]**p

    # Label initial points
    for x in initials:
        graph.label(x)

    # Run s2 algorithm 
    for j in range(budget):
        x = s2query_weighted(graph, weight_dictionary)

        graph.label(x)

    return graph.queried
