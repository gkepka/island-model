import random


def copy_best_individuals(population, n):
    best = sorted(population, key=lambda x: x.fitness.values[0])
    return best[:n]


def copy_worst_individuals(population, n):
    worst = sorted(population, key=lambda x: x.fitness.values[0], reverse=True)
    return worst[:n]


def copy_random_individuals(population, n):
    return random.sample(population, n)


def remove_random_individuals(population, n):
    random.shuffle(population)
    chosen = population[:n]
    del population[:n]
    return chosen
