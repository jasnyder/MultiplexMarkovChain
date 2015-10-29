#!/usr/bin/env python
"""
File contains functions to compute counts for constructing Markov
chains for the dynamics of edges on a two-layer multiplex network.
"""


import numpy as np
import networkx as nx
import logging

#set up logs
logger = logging.getLogger("multiplex_markov_chain")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def node_counts(g1,g2):
    """
    Returns a dictionary of node-based transition counts.
    Currently, the state of a node is computed as a binary vector indicating
    above/below average activity in each layer.
    g1 = list of graphs describing the multiplex at one time step
    g2 = list of graphs describing the multiplex at the next time step
    """
    counts = dict()
    num_layers = len(g1)
    node_set = get_node_set(g1,g2,method="intersection")
    mean_degrees1 = [np.mean(nx.degree(g1[k]).values()) for k in range(num_layers)]
    mean_degrees2 = [np.mean(nx.degree(g2[k]).values()) for k in range(num_layers)]
    for node in node_set:
        s1 = tuple(int(nx.degree(g1[k])[node]>mean_degrees1[k]) for k in range(num_layers))
        s2 = tuple(int(nx.degree(g2[k])[node]>mean_degrees2[k]) for k in range(num_layers))
        if (s1,s2) in counts.keys():
            counts[(s1,s2)] += 1
        else:
            counts[(s1,s2)] = 1
    return counts


def get_node_set(g1,g2,method="union"):
    """
    Returns the set of nodes that have to be considered in counting
    transitions of the Markov chains.  The input for the keyword
    argument `method` controls the method used.
    g1 and g2 are n-tuples of graphs, where n is the number of layers
    g1[k] is a graph describing the k-th layer in the multiplex at time t
    g2[k] is the k-th layer at time t+1
    """
    num_layers = len(g1)
    nodes1 = set()
    nodes2 = set()
    for k in range(num_layers):
        for n in g1[k].nodes():
            nodes1.add(n)
        for n in g2[k].nodes():
            nodes2.add(n)
    if (method=="intersection"):
        node_set = list(nodes1 & nodes2)
    else:
        node_set = list(nodes1 | nodes2)
    
    return node_set
    

def get_counts(g1, g2, method):
    """
    Computes the counts for each transition from time (t) step to time
    (t+1) given networkx graph instances for the two time steps.

    Parameters
    -----------
    g1 : list of nx graph objects representing the multiplex at time t
    g2 : list of nx graph objects representing the multiplex at time (t+1)

    OR

    g1 : Multinet instance representing the multiplex at time t
    g2 : Multinet instance representing the multiplex at time (t+1)

    The output is a dictionary giving
    counts from time t to time (t+1), for each possible pair of joint states.
    non-existence of an edge is coded as 0 by default.

    method : When the set of nodes in g1 is not the same as g2, the
    `method` to be used to find a common set of nodes. Accepts two
    values union or intersect.

    Returns
    -------
    counts : dictionary of counts for the transitions
    
    """
    # get the set of nodes to iterate over
    node_set = get_node_set(g1, g2, method)
    num_layers = len(g1)
    # Now count the numbers for each transition
    counts = dict()
    if g1[0].is_directed():
        for node1 in node_set:
            for node2 in node_set: # loop over all ordered pairs
                # 'state' codes all interactions between node1 and node2 (both directions)
                # state[k] is a tuple indicating the state of the dyad in layer k. This tuple can be (0,0),(0,1),(1,0), or (1,1)
                prev_state = tuple((int(g1[k].has_edge(node1,node2)), int(g1[k].has_edge(node2,node1))) for k in range(num_layers))
                current_state = tuple((int(g2[k].has_edge(node1,node2)), int(g2[k].has_edge(node2,node1))) for k in range(num_layers))
                if ((prev_state,current_state) in counts.keys()):
                    counts[(prev_state,current_state)] += 1
                else:
                    counts[(prev_state,current_state)] = 1
    else:
        for index,node1 in enumerate(node_set):
            for node2 in node_set[index+1:]: # loop over all unordered pairs
                prev_state = tuple(int(g1[k].has_edge(node1,node2)) for k in range(num_layers))
                current_state = tuple(int(g2[k].has_edge(node1,node2)) for k in range(num_layers))
                if ((prev_state,current_state) in counts.keys()):
                    counts[(prev_state,current_state)] += 1
                else:
                    counts[(prev_state,current_state)] = 1
    return counts


def compute_counts_from_file(fname_edges, fname_nodes=None, method=None):
    """
    Get as inputs file path for edges of the graph. Returns a
    dictionary with counts indexed by the time steps

    Parameters
    -----------
    fname_edges : path to a csv file of edge information in the
    following format.
    Time,Node1,Node2,Edge-Type1,Edge-Type2

    Time: Integer indicating the time the pair of nodes
    
    Edge-Type1 : Binary value. 0 (1) indicates absence (presence) of
    an edge of type 1

    Edge-Type2 : Binary value. 0 (1) indicates absence (presence) of
    an edge of type 2

    The file could be an edge-list, i.e., a list of only the edges
    present in the network or a list of all possible node-pairs with
    information of edges that are absent.

    fname_nodes : optional, path to a csv file with node information
    with the following format.  Time,Node

    Assumptions:
    - The values for Time above is non-decreasing.
    - When there is a change, time increases by 1.

    method : method to use when the set of nodes between two time
    steps are not the same. The variable accepts the strings `union`
    or `intersection`.

    Returns
    --------
    counts : dictionary with time steps as key and the np.array of
    counts for the transitions as the value.
    
    """
    fEdges = open(fname_edges,"r")
    fEdges.readline()
    counts = {}
    prevTimeStep = None
    timeStepToProcess = None

    # When method is not specified, if node file is given use the
    # intersection between two time steps, else use the union.
    if method is None:
        if fname_nodes is None:
            method = "union"
        else:
            method = "intersection"
    

    if (fname_nodes is not None):
        fNodes = open(fname_nodes,"r")
        fNodes.readline()
        nodeLine = fNodes.readline()
        nodeLine = nodeLine.rstrip()
        if (nodeLine):
            timeStep_nodes, node = nodeLine.split(",")
        else:
            nodeLine = None    
    for line in fEdges:
        line = line.rstrip()
        edge = line.split(",")
        if (len(edge) != 5):
            logger.warning("Line not in proper format. Ignoring %s",line)
            continue
        timeStep, n1, n2, eA, eB = edge
        if (timeStep != prevTimeStep):
            if prevTimeStep is not None:
                if (timeStepToProcess is None):
                    timeStepToProcess = prevTimeStep
                else:
                    # There are two graphs that are built. Get counts for them.
                    logger.info("Getting counts for %s-->%s",g_old.graph['time'],g_new.graph['time'])
                    c = get_counts(g_old, g_new, method = method)
                    counts[timeStepToProcess] = c
                    timeStepToProcess = prevTimeStep
                #New time step has started. Assign old graphs to the new ones.
                g_old = g_new
                # Start building a new graph for the current time.
            g_new = nx.Graph(time=timeStep)
            #add nodes for this timeStep
            if fname_nodes is not None:
                while (nodeLine and timeStep_nodes == timeStep):
                    g_new.add_node(node)
                    nodeLine = fNodes.readline()
                    nodeLine = nodeLine.rstrip()
                    if (nodeLine):
                        timeStep_nodes, node = nodeLine.split(",")
                    else:
                        nodeLine = None

        #another edge in the graph, process and store the state
        # assuming inputs are "nice"
        g_new.add_nodes_from([n1,n2])
        try:
            edgeState = 2*int(eB) + int(eA)
        except:
            logger.error("Edge '%s' cannot produce an integer valued state. Please check the input.", line)
        if (g_new.has_edge(n1,n2) and g_new.edge[n1][n2]["state"] != edgeState):
            logger.warning("Graph already has edge %s--%s in state %s",n1,n2,g_new.edge[n1][n2]["state"])
        g_new.add_edge(n1,n2, state=edgeState)
        prevTimeStep = timeStep

    logger.info("Reached end of edge list")
    logger.info("Getting counts for %s-->%s",g_old.graph['time'],g_new.graph['time'])
    c = get_counts(g_old, g_new, method)
    counts[timeStepToProcess] = c
    fEdges.close()
    return counts