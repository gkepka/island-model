import math


# Returns sum of taxicab distances between individuals
def get_l1_diversity(population):
    def get_sum_of_differences(index):
        arr = sorted(map(lambda x: x[index], population))
        total = 0
        arr_sum = 0
        for i in range(len(arr)):
            total += (arr[i] * i - arr_sum)
            arr_sum += arr[i]
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
            if i < j:
                total += math.dist(population[i], population[j])

    return total
