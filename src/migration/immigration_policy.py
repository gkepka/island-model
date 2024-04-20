import random


def substitute_worst_individuals(population, immigrants):
    population.sort(key=lambda x: x.fitness.values[0])
    del population[len(population) - len(immigrants):]
    population.extend(immigrants)


def substitute_random_individuals(population, immigrants):
    random.shuffle(population)
    del population[len(population) - len(immigrants):]
    population.extend(immigrants)


def add_all_individuals(population, immigrants):
    population.extend(immigrants)
