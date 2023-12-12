import time
import math
import numpy
import random
import argparse
import statistics

from deap import base, creator, tools, benchmarks
from opfunu.cec_based.cec2014 import F92014


parser = argparse.ArgumentParser()
parser.add_argument("-m") # mutation probability
parser.add_argument("-s") # mutation - sigma
parser.add_argument("-c") # crossover operator
parser.add_argument("-a") # blend crossover - alpha
parser.add_argument("-e") # sbx crossover - eta
parser.add_argument("-u") # uniform crossover - probability of exchange
parser.add_argument("-k") # tournament selection - k
args = parser.parse_args()


def set_crossover(toolbox):
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

# not tuned parameters
IND_SIZE=100
POP_SIZE=100
GENERATIONS=500

# tuned parameters
MUTATION_PROB = float(args.m)
SIGMA = float(args.s)
K = int(args.k)


func = F92014(IND_SIZE)

def evaluate(individual):
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
    for i in range(IND_SIZE):
        diversity += get_sum_of_differences(i)
    return diversity

# Returns sum of Euclidean distances between individuals
def get_l2_diversity(population):
    total = 0
    for i in range(POP_SIZE):
        for j in range(POP_SIZE):
            if i < j: total += math.dist(population[i], population[j])
                
    return total


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, a=-100, b=100)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

set_crossover(toolbox)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=SIGMA, indpb=MUTATION_PROB)
toolbox.register("select", tools.selTournament, tournsize=K)
toolbox.register("evaluate", evaluate)
toolbox.register("diversity", get_l1_diversity)


class Logger():
    def __init__(self):
        self.diversity_file = open("diversity.csv", "w")
        self.fitness_file = open("fitness.csv", "w")
    
    def log_diversity(self, generation, diversity):
        self.diversity_file.write(f'"{generation}","{diversity}"\n')
        
    def log_fitness(self, generation, fitness):
        self.fitness_file.write(f'"{generation}","{fitness}"\n')
        
    def close(self):
        self.diversity_file.close()
        self.fitness_file.close()


def run(logger: Logger):
    pop = toolbox.population(n=POP_SIZE)

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(GENERATIONS):
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
        logger.log_diversity(g, diversity)
        
        # Log best fitness
        best_fitness = tools.selBest(pop, 1)[0].fitness.values[0]
        logger.log_fitness(g, best_fitness)
        

    return tools.selBest(pop, 1)[0]


if __name__ == "__main__":
    start_time = time.time()
    logger = Logger()
    solution = run(logger)
    logger.close()
    #print(solution)
    #print("Fitness: %s" % solution.fitness.values)
    print("Time to complete: %s s" % (time.time() - start_time))
    print(solution.fitness.values[0])
