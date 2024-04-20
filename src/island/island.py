import ray
import queue

from deap import tools
from threading import Condition

@ray.remote
class Island:
    def __init__(self, index, toolbox, migration_policy, logger_class):
        self.index = index
        self.toolbox = toolbox
        self.migration_policy = migration_policy
        self.logger = logger_class(index)
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

    def run(self, generations):
        pop = self.toolbox.population()

        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in range(generations):
            # Update stored population
            self.set_population(pop)
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

            for mutant in offspring:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            # Log current diversity
            self.logger.log_diversity(g, pop, self.toolbox.diversity)

            # Log best fitness
            best_fitness = tools.selBest(pop, 1)[0].fitness.values[0]
            self.logger.log_fitness(g, best_fitness)

            # Log population before migration
            self.logger.log_population(g, pop)

            # Perform migration
            self.migration_policy.send_emigrants(self, pop, g)
            self.migration_policy.receive_immigrants(self, pop, g)

            if len(pop) != 100:
                raise Exception("Missing individuals")

        return tools.selBest(pop, 1)[0]

    def close(self):
        self.logger.close()
