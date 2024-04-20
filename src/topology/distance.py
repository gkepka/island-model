import ray

from diversity import get_l1_diversity


@ray.remote
def get_distance_between_pop(population1, population2):
    joined = population1 + population2
    return get_l1_diversity(joined) - get_l1_diversity(population1) - get_l1_diversity(population2)
