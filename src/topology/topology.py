import ray
import networkx as nx

from .distance import get_distance_between_pop


class Topology:
    def get_graph(self):
        pass

    def neighbors(self, island_index):
        return [ray.get_actor(str(i)) for i in self.get_graph().neighbors(island_index)]

    def send_to_neighbors(self, island_index, emigrants):
        neighbors = self.neighbors(island_index)
        for n in neighbors:
            n.send.remote(emigrants)

    def indegree(self, island_index):
        graph = self.get_graph()
        if nx.is_directed(graph):
            return len(list(graph.predecessors(island_index)))
        else:
            return len(list(graph.neighbors(island_index)))


class StaticTopology(Topology):
    def __init__(self, graph):
        self.graph = graph

    def get_graph(self):
        return self.graph


class MaxDistanceMatchingTopology(Topology):
    def __init__(self, islands):
        self.islands = islands

    def get_graph(self):
        indices = list(range(self.islands))
        neighbors = [ray.get_actor(str(i)) for i in indices]
        populations = ray.get([n.get_population.remote() for n in neighbors])
        zipped = zip(indices, populations)
        pop_dict = {el[0]: el[1] for el in zipped}

        graph = nx.complete_graph(self.islands)
        joined_indices = [(i, j) for i in indices for j in indices if i < j]
        distances = ray.get([get_distance_between_pop.remote(pop_dict[x[0]], pop_dict[x[1]]) for x in joined_indices])
        for index in range(len(joined_indices)):
            i, j = joined_indices[index]
            graph[i][j]['weight'] = distances[index]

        matching = nx.max_weight_matching(graph)

        return nx.from_edgelist(matching)

    def indegree(self, island_index):
        return 1
