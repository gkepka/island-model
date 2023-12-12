import ray
import time
import math
import numpy
import queue
import random
import argparse
import networkx as nx

from deap import base, creator, tools, benchmarks
from opfunu.cec_based.cec2014 import F92014


class Logger():
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
    weights = weights=(-1.0,)
    def __init__(self):
        super().__init__()


class Individual(list):
    def __init__(self, initialValues):
        super().__init__(initialValues)
        self.fitness = FitnessMin()


class Topology():
    def __init__(self, graph):
        self.graph = graph
        
    def neighbors(self, island):
        return [ray.get_actor(str(i)) for i in self.graph.neighbors(island)]
    
    def neighbors_count(self, island):
        return len(self.graph[island])
    
    def send_to_neighbors(self, island, message):
        neighbors = self.neighbors(island)
        for n in neighbors:
            n.send.remote(message)

@ray.remote
class Island():
    def __init__(self, toolbox, topology, index):
        self.toolbox = toolbox
        self.logger = Logger(index)
        self.topology = topology
        self.index = index
        self.queue = queue.Queue()
        
    def send(self, message):
        self.queue.put(message, timeout=10)
        
    def receive(self, count):
        msgs = []
        for i in range(count):
            msgs.append(self.queue.get(timeout=10))
        return msgs
        
    def run(self, generations, migration_interval):
        pop = toolbox.population()

        # Evaluate the entire population
        fitnesses = map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in range(generations):
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
            if g % migration_interval == 0:
                # Get emigrants
                emigrants = toolbox.get_emigrants(pop)
                self.topology.send_to_neighbors(self.index, emigrants)
                received = self.receive(self.topology.neighbors_count(self.index))
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

def clone_best_individuals(population, n):
    best = sorted(population, key = lambda x: x.fitness.values[0], reverse=True)
    return best[:n]

def substitute_worst_individuals(population, imigrants):
    population.sort(key = lambda x: x.fitness.values[0], reverse=True)
    del population[len(population)-len(imigrants):]
    population.extend(imigrants)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m") # mutation probability
    parser.add_argument("-s") # mutation - sigma
    parser.add_argument("-c") # crossover operator
    parser.add_argument("-a") # blend crossover - alpha
    parser.add_argument("-e") # sbx crossover - eta
    parser.add_argument("-u") # uniform crossover - probability of exchange
    parser.add_argument("-k") # tournament selection - k
    args = parser.parse_args()
    
    # not tuned parameters
    IND_SIZE=100
    POP_SIZE=100
    GENERATIONS=500
    NUM_OF_ISLANDS=8
    NUM_OF_MIGRANTS=3
    MIGRATION_INTERVAL=50

    # tuned parameters
    MUTATION_PROB = float(args.m)
    SIGMA = float(args.s)
    K = int(args.k)
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, a=-100, b=100)
    toolbox.register("individual", tools.initRepeat, Individual, toolbox.attr_float, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=POP_SIZE)

    register_crossover(args, toolbox)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=SIGMA, indpb=MUTATION_PROB)
    toolbox.register("select", tools.selTournament, tournsize=K)
    toolbox.register("evaluate", evaluate, func=F92014(IND_SIZE))
    toolbox.register("diversity", get_l1_diversity)
    
    toolbox.register("get_emigrants", clone_best_individuals, n=NUM_OF_MIGRANTS)
    toolbox.register("accept_imigrants", substitute_worst_individuals)
    
    start_time = time.time()

    graph = nx.cycle_graph(NUM_OF_ISLANDS)
    topology = Topology(graph)
    islands = [Island.options(name=str(i), max_concurrency=2).remote(toolbox, topology, i) for i in range(NUM_OF_ISLANDS)]    
    solutions = ray.get([island.run.remote(GENERATIONS, MIGRATION_INTERVAL) for island in islands])
    end_time = time.time()

    for island in islands:
        island.close.remote()
    
    for solution in solutions:
        print(solution.fitness.values[0])
        
    ordered = sorted(solutions, key = lambda x: x.fitness.values[0])
    print(f"Best: {ordered[0].fitness.values[0]}")

    print("Time to complete: %s s" % (end_time - start_time))
 
