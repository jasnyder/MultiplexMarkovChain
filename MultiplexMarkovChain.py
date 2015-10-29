#!/usr/bin/env python
"""

Class that constructs a MultiplexMarkovchain given the counts. Also
has methods to get the null model and the parameters of the
probability distribution associated with the Markov chain. Currently
handles only 4-state Markov chains (i.e. two layer networks).

"""
from __future__ import division
import numpy as np
import networkx as nx
from warnings import warn


class MarkovChain(nx.DiGraph):
    """
    Instances of this class contain NetworkX DiGraph objects, whose edges have
    attributes "counts" and "params", giving observed counts of transitions
    and inferred transition probabilities, respectively.

    A class that computes properties of a Markov chain such as the
    transition parameters and standard deviation. The class assumes a
    uniform prior for the distribution of transition parameters.

    Each state of the Markov chain has a Dirichlet (multivariate beta)
    distribution whose variates are the transition parameters. The
    transition parameters are estimated as the mean of the
    corresponding Dirichlet distribution.

    Parameters
    ----------

    counts : Counts for each transition of the Markov chain. Given as a dictionary
    in the form counts = {(from_state,to_state):count}. For a Multiplex Markov Chain,
    states themselves will be n-tuples (n = number of layers) whose entries
    are the states in the individual layers

    num_transitions : The number of transitions in the Markov chain. The number
    of states of the Markov chain is sqrt(num_transitions). len(counts) =
    num_transitions

    state_totals : The total of counts leaving from the particular
    state of the Markov chain. len(state_totals) is equal to the
    number of states of the Markov chain. Can be accessed using the
    method `get_state_totals`.

    params : The average value of the probability of the transitions
    assuming a uniform prior. Stored as a dictionary whose keys are
    tuples (from_state,to_state). Can be accessed using method 'get_parameters'

    std : The standard deviations of the variates of the Dirichlet
    distributions associated with the Markov chain. Stored as a dictionary
    whose keys are tuples (from_state,to_state). Can be accessed using
    method 'get_std_dev'

    """

    def __init__(self, counts):
        if type(counts) is not dict:
            raise AssertionError("counts must be presented as a dictionary: {transition: count}")
        elif any([type(k) is not tuple for k in counts.keys()]):
            raise AssertionError("keys of 'counts' must be tuples: (from_state, to_state)")
        elif any([len(k) != 2 for k in counts.keys()]):
            raise AssertionError("keys of 'counts' must be length-2 tuples: (from_state, to_state)")
        elif any([type(v) is not int for v in counts.values()]):
            raise AssertionError("values of 'counts' must be integers")
        self.counts = counts
        self.MC = nx.DiGraph()
        for (from_state,to_state) in counts.keys():
            self.MC.add_edge(from_state,to_state, count = counts[(from_state,to_state)])
        #pad 'counts' with zeros and add all missing edges to self.MC
        for from_state in self.MC.nodes():
            for to_state in self.MC.nodes():
                if ((from_state,to_state) not in counts.keys()):
                    self.MC.add_edge(from_state,to_state,count=0)
                    counts[(from_state,to_state)] = 0
        self.params = None # probability of transitions
        self.std = None # std. associated with the transitions
        self.state_totals = None #total number of transitions leaving a state
        self.num_transitions = (self.MC.number_of_nodes())**2

    def compute_prob_params(self,counts):
        """
        Given counts returns the mean, std. dev. for every transition
        and normalization constant for each state, and attatches these 
        values to the edges of the MC.

        It also packs the values into a dictionary (self.params, self.std)
        whose keys are edges (ordered pairs of states)
        """
        num_transitions  = self.num_transitions
        l = self.MC.number_of_nodes()
        totals = dict()
        for from_state in self.MC.nodes():
            tot = sum([counts[(from_state,to_state)] if ((from_state,to_state) in counts.keys()) else 0 for to_state in self.MC.nodes()])
            totals[from_state] = tot
            if tot > 0:
                for to_state in self.MC.nodes():
                    # mean and std. dev of the corresponding beta distribution
                    p = (counts[(from_state,to_state)]+1)/(tot+l) if ((from_state,to_state) in counts.keys()) else 1/(tot+l)
                    self.MC[from_state][to_state]['mu'] = p
                    self.MC[from_state][to_state]['sigma'] = np.sqrt(p*(1-p)/(tot+ (l+1)))
            else:
                for to_state in self.MC.nodes():
                    p = 1/(tot+l)
                    self.MC[from_state][to_state]['mu'] = p
                    self.MC[from_state][to_state]['sigma'] = np.sqrt(p*(1-p)/(tot+ (l+1)))

        self.params = {(fs,ts):self.MC[fs][ts]['mu'] for (fs,ts) in self.MC.edges()}
        self.std = {(fs,ts):self.MC[fs][ts]['sigma'] for (fs,ts) in self.MC.edges()}
        self.state_totals = totals


    def get_parameters(self):
        if self.params is None:
            self.compute_prob_params(self.counts)
        return self.params

    def get_std_dev(self):
        if self.std is None:
            self.compute_prob_params(self.counts)
        return self.std

    def get_state_totals(self):
        if self.state_totals is None:
            self.compute_prob_params(self.counts)
        return self.state_totals



def _is_power_of_2(x):
    return (x & (x-1) == 0 and x != 0)


class MultiplexMarkovChain(MarkovChain):
    """
    Class inherits from MarkovChain. In addition, the class builds a
    null model for detecting `dynamical spillover` (Insert ref. to
    paper).

    To build an instance of this class, you must provide a dictionary of
    counts, formatted as for the MarkovChain class. Moreover, states must
    have the form (layer1,layer2,...,layer_n), specifying the states of
    the edge in the n layers of the multiplex.

    In particular, keys of the dictionary 'counts' must have the form:
    ((layer1, layer2, ..., layer_n),(layer1', layer2', ..., layer_n'))

    Parameters
    ----------

    null_components : a list with a dictionary for each layer of the
    multiplex network. The dictionary has two items:

        counts : the total counts associated with a Markov chain
        describing the edge dynamics on a particular layer

        MC : the MarkovChain initialized with the above counts.


    null_prob : transition parameters for the null model.

    null_std : standard deviation associated with the transition
    parameters of the null model.

    See Also
    ---------
    MarkovChain

    """


    def __init__(self, counts):
        '''
        num_transitions = len(counts)
        #check if the num_transitions is a power of 2.
        if not _is_power_of_2(num_transitions):
            raise AssertionError("Length of counts is not a power of 2.")
        '''
        MarkovChain.__init__(self, counts)
        self.counts = counts
        self.num_layers = len(counts.keys()[0][0])
        self.null_components = None
        self.null_prob = None
        self.null_std = None

    def _compute_null_counts(self, counts):
        """
        This function computes counts for the null model.
        """
        num_layers = self.num_layers
        null_counts = [dict() for i in range(num_layers)]
        for (joint_fs,joint_ts) in counts.keys():
            for layer in range(num_layers):
                if (joint_fs[layer],joint_ts[layer]) in null_counts[layer].keys():
                    null_counts[layer][(joint_fs[layer],joint_ts[layer])] += counts[(joint_fs,joint_ts)]
                else:
                    null_counts[layer][(joint_fs[layer],joint_ts[layer])] = counts[(joint_fs,joint_ts)]

        self.null_components = [{'counts':null_counts[i]} for i in range(num_layers)]

    def compute_prob_null_components(self):
        """
        Initializes a MarkovChain for each layer of the multiplex that
        describes the evolution of edges on that layer independent of
        the other layers.
        """
        if self.null_components is None:
            self._compute_null_counts(self.counts)
        for component in self.null_components:
            component["MC"] = MarkovChain(component["counts"])


    def compute_null_components(self):
        """
        Computes the components of the null model. For a 4-state MC,
        they are two 2-state MCs.
        """
        self._compute_null_counts(self.counts)
        self.compute_prob_null_components()

    def compute_null_prob_std(self):
        """
        Computes the null probability using the null components. When
        computing the standard deviation the method approximates the
        beta distributions as a Gaussian distributions.
        """
        num_transitions = self.num_transitions
        num_layers = self.num_layers
        pnull = dict()
        std_null = dict()
        #If the Gaussian approximation is not justified warn the user
        state_totals = self.get_state_totals()
        if (np.any(state_totals < 100)):
            warn("Some of the state totals are less than 100. Gaussian approximation may not be justified.")
        
        component_params = [self.null_components[k]['MC'].get_parameters() for k in range(num_layers)]
        component_std_dev = [self.null_components[k]['MC'].get_std_dev() for k in range(num_layers)]
        pnull = {(fs,ts):np.prod([component_params[k][(fs[k],ts[k])] for k in range(num_layers)]) for (fs,ts) in self.MC.edges()}
        std_null = {(fs,ts):
        pnull[(fs,ts)]*np.sqrt(
            np.sum(
                [(component_std_dev[k][(fs[k],ts[k])]/component_params[k][(fs[k],ts[k])])**2 for k in range(num_layers)]
                )
            ) for (fs,ts) in self.MC.edges()}
        self.null_prob = pnull
        self.null_std = std_null


    def get_null_prob(self):
        """
        Return the probability for each transition
        of the null model. Computes the null model the first time this
        function is called.
        """
        if self.null_prob is None:
            if self.null_components is None:
                self.compute_null_components()

            self.compute_null_prob_std()
        return self.null_prob


    def get_null_std_dev(self):
        """
        Returns the std dev. associated with the probability
        distribution of the transition parameters of the null model.
        """
        if self.null_std is None:
            if self.null_components is None:
                self.compute_null_components()
            self.compute_null_prob_std()
        return self.null_std

    def differs_from_null(self):
        """
        Returns a dictionary whose keys are transitions
        and whose values are abs(null_prob - prob)/(std_null + std)
        """


        if self.null_prob is None:
            if self.null_components is None:
                self.compute_null_components()
            self.compute_null_prob_std()
        if self.params is None:
            self.compute_prob_params(self.counts)

        return {k:(self.params[k]-self.null_prob[k])/(self.std[k]+self.null_std[k]) for k in self.params.keys()}
