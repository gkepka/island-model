import ray
import time
import numpy
import opfunu
import random
import logging
import argparse

from deap import base, tools

from island import Island
from logger import Logger, NoOpLogger
from diversity import get_l1_diversity
from topology import MaxDistanceMatchingTopology
from migration import MigrationPolicy, Synchronizer, add_all_individuals, remove_random_individuals


class FitnessMin(base.Fitness):
    weights = (-1.0,)

    def __init__(self):
        super().__init__()


class Individual(list):
    def __init__(self, initial_values):
        super().__init__(initial_values)
        self.fitness = FitnessMin()


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m")  # mutation probability
    parser.add_argument("-s")  # mutation - sigma
    parser.add_argument("-c")  # crossover operator
    parser.add_argument("-a")  # blend crossover - alpha
    parser.add_argument("-e")  # sbx crossover - eta
    parser.add_argument("-u")  # uniform crossover - probability of exchange
    parser.add_argument("-k")  # tournament selection - k
    parser.add_argument("-f")  # test function
    parser.add_argument('--logging', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    # not tuned parameters
    IND_SIZE = 100
    POP_SIZE = 100
    GENERATIONS = 500
    NUM_OF_ISLANDS = 8
    NUM_OF_MIGRANTS = 50
    MIGRATION_INTERVAL = 50

    # tuned parameters
    MUTATION_PROB = float(args.m)
    SIGMA = float(args.s)
    K = int(args.k)

    # configuration
    FUNCTION_NAME = args.f
    SHOULD_LOG = args.logging

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, a=-100, b=100)
    toolbox.register("individual", tools.initRepeat, Individual, toolbox.attr_float, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=POP_SIZE)

    func = opfunu.get_functions_by_classname(args.f)[0](IND_SIZE)

    register_crossover(args, toolbox)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=SIGMA, indpb=MUTATION_PROB)
    toolbox.register("select", tools.selTournament, tournsize=K)
    toolbox.register("evaluate", evaluate, func=func)
    toolbox.register("diversity", get_l1_diversity)

    start_time = time.time()
    ray.init(logging_level=logging.ERROR, log_to_driver=False)

    #graph = nx.hypercube_graph(3)
    #graph = nx.convert_node_labels_to_integers(graph)
    #topology = StaticTopology(graph)

    topology = MaxDistanceMatchingTopology(NUM_OF_ISLANDS)

    synchronizer = Synchronizer.options(max_concurrency=NUM_OF_ISLANDS).remote(NUM_OF_ISLANDS)
    migration_policy = MigrationPolicy(topology, NUM_OF_MIGRANTS, MIGRATION_INTERVAL, remove_random_individuals,
                                       add_all_individuals, synchronizer=synchronizer, cached=True)

    islands = [Island.options(name=str(index), max_concurrency=2).remote(index, toolbox, migration_policy, Logger if SHOULD_LOG else NoOpLogger) for index in
               range(NUM_OF_ISLANDS)]
    solutions = ray.get([island.run.remote(GENERATIONS) for island in islands])
    end_time = time.time()

    for island in islands:
        island.close.remote()

    ordered = sorted(solutions, key=lambda x: x.fitness.values[0])
    print(ordered[0].fitness.values[0])
    # print("Time to complete: %s s" % (end_time - start_time))
