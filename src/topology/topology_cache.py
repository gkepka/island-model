import ray

from threading import Lock
from topology import StaticTopology


@ray.remote
class RemoteTopologyCache:
    # cached_topology should be a dynamic topology, like MaxDistanceMatchingTopology
    def __init__(self, cached_topology):
        self.cached_topology = cached_topology
        self.lock = Lock()
        self.topology_cache = {}

    def get_cached_topology(self, generation):
        self.lock.acquire(timeout=10)
        try:
            if generation not in self.topology_cache:
                self.topology_cache[generation] = StaticTopology(self.cached_topology.get_graph())
            return self.topology_cache[generation]
        finally:
            self.lock.release()
