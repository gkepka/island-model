import ray

from topology import RemoteTopologyCache


class MigrationPolicy:
    def __init__(self, topology, migration_size, migration_interval, select_emigrants, accept_immigrants,
                 synchronizer=None, cached=False):
        self.migration_size = migration_size
        self.migration_interval = migration_interval
        self.select_emigrants = select_emigrants
        self.accept_immigrants = accept_immigrants
        if synchronizer is not None:
            self.synchronizer = synchronizer
            self.synchronized = True
        else:
            self.synchronized = False
        if cached and not self.synchronized:
            raise Exception("Cannot cache topology in non-synchronized model")
        if cached:
            self.topology_cache = RemoteTopologyCache.remote(topology)
            self.get_topology = lambda generation: ray.get(self.topology_cache.get_cached_topology.remote(generation))
        else:
            self.get_topology = lambda generation: topology

    def send_emigrants(self, island, population, generation):
        if generation < self.migration_interval or generation % self.migration_interval != 0:
            return
        if self.synchronized:
            ray.get(self.synchronizer.wait.remote())
        emigrants = self.select_emigrants(population, self.migration_size)
        self.get_topology(generation).send_to_neighbors(island.index, emigrants)

    def receive_immigrants(self, island, population, generation):
        if self.synchronized and generation >= self.migration_interval and generation % self.migration_interval == 0:
            received = island.receive_sync(self.get_topology(generation).indegree(island.index))
            for immigrants in received:
                self.accept_immigrants(population, immigrants)
        elif not self.synchronized:
            received = island.receive_async()
            for immigrants in received:
                self.accept_immigrants(population, immigrants)
