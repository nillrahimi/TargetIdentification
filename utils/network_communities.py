import networkx as nx


class NetworkCommunities:
    def __init__(self, G: nx.classes.graph.Graph = None):
        self._graph = G
        self._community = None

    def communities_girvan_newman(self):
        """
        Apply Girvan-Newman algorithm
        :return:
        """
        communities_generator = nx.community.girvan_newman(self._graph)
        next_level_communities = next(communities_generator)
        self._community = sorted(map(sorted, next_level_communities))
        return self._community

    def communities_louvain_algorithm(self):
        """
        Apply Louvain algorithm
        :param seed: The seed value for the Louvain algorithm
        :return: The community structure
        """
        communities_generator = nx.community.louvain_communities(self._graph)
        self._community = sorted(map(sorted, communities_generator))
        return self._community

    def communities_label_propagation(self):
        """
        Apply label propagation algorithm
        :param k:
        :return:
        """
        communities_generator = (
            nx.community.label_propagation.label_propagation_communities(self._graph)
        )
        self._community = sorted(map(sorted, communities_generator))
        return self._community

    def communities_asyn_fluidc(self, k: int = 8):
        communities_generator = nx.community.asyn_fluidc(self._graph, k)
        next_level_communities = next(communities_generator)
        self._community = sorted(map(sorted, next_level_communities))
        return self._community

    def modularities(self):
        return nx.community.modularity(self._graph, self._community)
