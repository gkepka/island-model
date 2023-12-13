import ray
import time
import math
import numpy
import queue
import random
import argparse
import networkx as nx

from threading import Condition, Barrier, Lock
from opfunu.cec_based.cec2014 import F92014
from deap import base, tools


class Logger:
    def __init__(self, suffix):
        self.diversity_file = open(f"logs/diversity_{suffix}.csv", "w")
        self.fitness_file = open(f"logs/fitness_{suffix}.csv", "w")

    def log_diversity(self, generation, diversity):
        self.diversity_file.write(f'"{generation}","{diversity}"\n')

    def log_fitness(self, generation, fitness):
        self.fitness_file.write(f'"{generation}","{fitness}"\n')

    def close(self):
        self.diversity_file.close()
        self.fitness_file.close()


class FitnessMin(base.Fitness):
    weights = (-1.0,)

    def __init__(self):
        super().__init__()


class Individual(list):
    def __init__(self, initial_values):
        super().__init__(initial_values)
        self.fitness = FitnessMin()


@ray.remote
class Synchronizer:
    def __init__(self, islands):
        self.barrier = Barrier(islands)

    def wait(self):
        self.barrier.wait(timeout=60)


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


@ray.remote
class RemoteTopologyCache:
    # cached_topology should be a dynamic topology, like MaxDistanceMatchingTopology
    def __init__(self, cached_topology):
        self.cached_topology = cached_topology
        self.lock = Lock()
        self.topology_cache = {}
        
    def get_cached_topology(self, generation):
        self.lock.acquire(timeout=10)
        try:
            if generation not in self.topology_cache:
                self.topology_cache[generation] = StaticTopology(self.cached_topology.get_graph())
            return self.topology_cache[generation]
        finally:
            self.lock.release()


class MigrationPolicy:
    def __init__(self, topology, migration_size, migration_interval, select_emigrants, accept_imigrants, synchronizer=None, cached=False):
        self.migration_size = migration_size
        self.migration_interval = migration_interval
        self.select_emigrants = select_emigrants
        self.accept_imigrants = accept_imigrants
        if synchronizer is not None:
            self.synchronizer = synchronizer
            self.synchronized = True
        else:
            self.synchronized = False
        if cached and not self.synchronized:
            raise Exception("Cannot cache topology in non-synchronized model")
        if cached:
            self.topology_cache = RemoteTopologyCache.remote(topology)
            self.get_topology = lambda generation: ray.get(self.topology_cache.get_cached_topology.remote(generation))
        else:
            self.get_topology = lambda generation: topology
            
    def send_emigrants(self, island, population, generation):
        if generation < self.migration_interval or generation % self.migration_interval != 0:
            return
        if self.synchronized:
            ray.get(self.synchronizer.wait.remote())
        emigrants = self.select_emigrants(population, self.migration_size)
        self.get_topology(generation).send_to_neighbors(island.index, emigrants)
        
    def receive_imigrants(self, island, population, generation):
        if self.synchronized and generation >= self.migration_interval and generation % self.migration_interval == 0:
            received = island.receive_sync(self.get_topology(generation).indegree(island.index))
            for imigrants in received:
                self.accept_imigrants(population, imigrants)
        elif not self.synchronized:
            received = island.receive_async()
            for imigrants in received:
                self.accept_imigrants(population, imigrants)


@ray.remote
class Island:
    def __init__(self, index, toolbox, migration_policy):
        self.index = index
        self.toolbox = toolbox
        self.migration_policy = migration_policy
        self.logger = Logger(index)
        self.queue = queue.Queue()
        self.cv = Condition()

    def send(self, message):
        self.queue.put(message, timeout=10)

    def receive_async(self):
        msgs = []
        while True:
            try:
                msgs.append(self.queue.get(block=False))
            except queue.Empty:
                break
        return msgs

    def receive_sync(self, count):
        msgs = []
        for i in range(count):
            msgs.append(self.queue.get(timeout=10))

        return msgs

    def get_population(self):
        with self.cv:
            self.cv.wait_for(lambda: hasattr(self, "population"), timeout=10)
            return self.population

    def set_population(self, pop):
        with self.cv:
            self.population = pop[:]
            self.cv.notify()

    def run(self, generations, migration_interval):
        pop = toolbox.population()

        # Evaluate the entire population
        fitnesses = map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in range(generations):
            # Update stored population
            self.set_population(pop)
            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

            for mutant in offspring:
                toolbox.mutate(mutant)
                del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            # Log current diversity
            diversity = toolbox.diversity(pop)
            self.logger.log_diversity(g, diversity)

            # Log best fitness
            best_fitness = tools.selBest(pop, 1)[0].fitness.values[0]
            self.logger.log_fitness(g, best_fitness)

            # Perform migration
            migration_policy.send_emigrants(self, pop, g)
            migration_policy.receive_imigrants(self, pop, g)

            if len(pop) != 100:
                raise Exception("Missing individuals")

        return tools.selBest(pop, 1)[0]

    def close(self):
        self.logger.close()


def register_crossover(args, toolbox):
    name = args.c
    if name == "one_point":
        toolbox.register("mate", tools.cxOnePoint)
    elif name == "two_point":
        toolbox.register("mate", tools.cxTwoPoint)
    elif name == "uniform":
        toolbox.register("mate", tools.cxUniform, indpb=float(args.u))
    elif name == "blend":
        toolbox.register("mate", tools.cxBlend, alpha=float(args.a))
    elif name == "sbx":
        toolbox.register("mate", tools.cxSimulatedBinary, eta=int(args.e))
    else:
        raise Exception("Unknown crossover type: " + name)


def evaluate(individual, func):
    arr = numpy.asarray(individual)
    return func.evaluate(arr),


# Returns sum of taxicab distances between individuals
def get_l1_diversity(population):
    def get_sum_of_differences(index):
        arr = sorted(map(lambda x: x[index], population))
        total = 0
        arrSum = 0
        for i in range(len(arr)):
            total += (arr[i] * i - arrSum)
            arrSum += arr[i]
        return total

    diversity = 0
    ind_size = len(population[0])
    for i in range(ind_size):
        diversity += get_sum_of_differences(i)
    return diversity


# Returns sum of Euclidean distances between individuals
def get_l2_diversity(population):
    pop_size = len(population)
    total = 0
    for i in range(pop_size):
        for j in range(pop_size):
            if i < j: total += math.dist(population[i], population[j])

    return total


def select_best_individuals(population, n):
    best = sorted(population, key=lambda x: x.fitness.values[0])
    return best[:n]


def select_worst_individuals(population, n):
    worst = sorted(population, key=lambda x: x.fitness.values[0], reverse=True)
    return worst[:n]


def select_random_individuals(population, n):
    return random.sample(population, n)


def substitute_worst_individuals(population, imigrants):
    population.sort(key=lambda x: x.fitness.values[0])
    del population[len(population) - len(imigrants):]
    population.extend(imigrants)


def substitute_random_individuals(population, imigrants):
    random.shuffle(population)
    del population[len(population) - len(imigrants):]
    population.extend(imigrants)


@ray.remote
def get_distance_between_pop(population1, population2):
    joined = population1 + population2
    return get_l1_diversity(joined) - get_l1_diversity(population1) - get_l1_diversity(population2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m")  # mutation probability
    parser.add_argument("-s")  # mutation - sigma
    parser.add_argument("-c")  # crossover operator
    parser.add_argument("-a")  # blend crossover - alpha
    parser.add_argument("-e")  # sbx crossover - eta
    parser.add_argument("-u")  # uniform crossover - probability of exchange
    parser.add_argument("-k")  # tournament selection - k
    args = parser.parse_args()

    # not tuned parameters
    IND_SIZE = 100
    POP_SIZE = 100
    GENERATIONS = 500
    NUM_OF_ISLANDS = 32
    NUM_OF_MIGRANTS = 5
    MIGRATION_INTERVAL = 50

    # tuned parameters
    MUTATION_PROB = float(args.m)
    SIGMA = float(args.s)
    K = int(args.k)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, a=-100, b=100)
    toolbox.register("individual", tools.initRepeat, Individual, toolbox.attr_float, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=POP_SIZE)

    register_crossover(args, toolbox)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=SIGMA, indpb=MUTATION_PROB)
    toolbox.register("select", tools.selTournament, tournsize=K)
    toolbox.register("evaluate", evaluate, func=F92014(IND_SIZE))
    toolbox.register("diversity", get_l1_diversity)

    start_time = time.time()

    graph = nx.hypercube_graph(5)
    graph = nx.convert_node_labels_to_integers(graph)
    topology = MaxDistanceMatchingTopology(NUM_OF_ISLANDS)
    #topology = StaticTopology(graph)
    synchronizer = Synchronizer.options(max_concurrency=NUM_OF_ISLANDS).remote(NUM_OF_ISLANDS)
    migration_policy = MigrationPolicy(topology, NUM_OF_MIGRANTS, MIGRATION_INTERVAL, select_best_individuals, substitute_worst_individuals, synchronizer=synchronizer, cached=True)
    
    islands = [Island.options(name=str(index), max_concurrency=2).remote(index, toolbox, migration_policy) for index in range(NUM_OF_ISLANDS)]
    solutions = ray.get([island.run.remote(GENERATIONS, MIGRATION_INTERVAL) for island in islands])
    end_time = time.time()

    for island in islands:
        island.close.remote()

    for solution in solutions:
        print(solution.fitness.values[0])

    ordered = sorted(solutions, key=lambda x: x.fitness.values[0])
    print(f"Best: {ordered[0].fitness.values[0]}")
    print("Time to complete: %s s" % (end_time - start_time))
