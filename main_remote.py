import ray
import time
import math
import numpy
import queue
import random
import argparse
import networkx as nx

from threading import Condition, Barrier
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


class Topology:
    def __init__(self, graph):
        self.graph = graph

    def neighbors(self, island):
        return [ray.get_actor(str(i)) for i in self.graph.neighbors(island.index)]

    def neighbors_count(self, island):
        return len(self.graph[island.index])

    def send_to_neighbors(self, island, emigrants):
        neighbors = self.neighbors(island)
        for n in neighbors:
            n.send.remote(emigrants)


class MaxDistanceTopology(Topology):
    def __init__(self, islands):
        super().__init__(nx.complete_graph(islands))
        self.islands = islands

    def neighbors(self, island):
        neighbors = [ray.get_actor(str(i)) for i in self.graph.neighbors(island.index)]
        populations = ray.get([n.get_population.remote() for n in neighbors])
        distances = [get_distance_between_pop(island.get_population(), p) for p in populations]
        with_neighbors = sorted(zip(distances, neighbors), key=lambda x: x[0], reverse=True)
        return [with_neighbors[0][1]]

    def neighbors_count(self, island):
        return 1


class MaxDistanceMatchingTopology(Topology):
    def __init__(self, islands):
        super().__init__(nx.complete_graph(islands))

    def neighbors(self, island):
        indices = list(self.graph.neighbors(island.index))
        neighbors = [ray.get_actor(str(i)) for i in indices]
        populations = ray.get([n.get_population.remote() for n in neighbors])
        zipped = zip(indices, populations)
        pop_dict = {el[0]: el[1] for el in zipped}
        pop_dict[island.index] = island.get_population()

        indices.append(island.index)
        for i in indices:
            for j in indices:
                if i < j: self.graph[i][j]['weight'] = get_distance_between_pop(pop_dict[i], pop_dict[j])

        matching = nx.max_weight_matching(self.graph)
        edge = list(filter(lambda x: x[0] == island.index or x[1] == island.index, matching))[0]
        neighbor_index = edge[0] if edge[1] == island.index else edge[1]

        return [ray.get_actor(str(neighbor_index))]

    def neighbors_count(self, island):
        return 1


@ray.remote
class Synchronizer:
    def __init__(self, islands):
        self.barrier = Barrier(islands)

    def wait(self):
        self.barrier.wait(timeout=10)


@ray.remote
class Island:
    def __init__(self, toolbox, topology, index, synchronizer):
        self.toolbox = toolbox
        self.logger = Logger(index)
        self.topology = topology
        self.index = index
        self.synchronizer = synchronizer
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
            if g >= migration_interval and g % migration_interval == 0:
                # Synchronize with other islands:
                ray.get(self.synchronizer.wait.remote())
                # Send emigrants
                emigrants = toolbox.select_emigrants(pop)
                self.topology.send_to_neighbors(self, emigrants)
                # Receive imigrants
                received = self.receive_sync(self.topology.neighbors_count(self))
                for imigrants in received:
                    toolbox.accept_imigrants(pop, imigrants)

            # Receive imigrants
            received = self.receive_async()
            for imigrants in received:
                toolbox.accept_imigrants(pop, imigrants)

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
    NUM_OF_ISLANDS = 8
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

    toolbox.register("select_emigrants", select_best_individuals, n=NUM_OF_MIGRANTS)
    toolbox.register("accept_imigrants", substitute_worst_individuals)

    start_time = time.time()

    graph = nx.hypercube_graph(3)
    graph = nx.convert_node_labels_to_integers(graph)
    topology = Topology(graph)
    synchronizer = Synchronizer.options(max_concurrency=NUM_OF_ISLANDS).remote(NUM_OF_ISLANDS)
    islands = [Island.options(name=str(i), max_concurrency=2).remote(toolbox, topology, i, synchronizer) for i in
               range(NUM_OF_ISLANDS)]
    solutions = ray.get([island.run.remote(GENERATIONS, MIGRATION_INTERVAL) for island in islands])
    end_time = time.time()

    for island in islands:
        island.close.remote()

    for solution in solutions:
        print(solution.fitness.values[0])

    ordered = sorted(solutions, key=lambda x: x.fitness.values[0])
    print(f"Best: {ordered[0].fitness.values[0]}")
    print("Time to complete: %s s" % (end_time - start_time))
