class Logger:
    def __init__(self, index):
        self.diversity_file = open(f"logs/diversity_{index}.csv", "w")
        self.fitness_file = open(f"logs/fitness_{index}.csv", "w")
        self.population_file = open(f"logs/population_{index}.csv", "w")

    def log_diversity(self, generation, population, diversity_function):
        diversity = diversity_function(population)
        self.diversity_file.write(f'"{generation}","{diversity}"\n')

    def log_fitness(self, generation, fitness):
        self.fitness_file.write(f'"{generation}","{fitness}"\n')

    def log_population(self, generation, population):
        if generation % 50 == 0:
            for ind_index, individual in enumerate(population):
                for gene_index, gene in enumerate(individual):
                    self.population_file.write(f'"{generation}","{ind_index}","{gene_index}","{gene}"\n')

    def close(self):
        self.diversity_file.close()
        self.fitness_file.close()
        self.population_file.close()


class NoOpLogger:
    def __init__(self, index):
        pass

    def log_diversity(self, generation, population, diversity_function):
        pass

    def log_fitness(self, generation, fitness):
        pass

    def log_population(self, generation, population):
        pass

    def close(self):
        pass
