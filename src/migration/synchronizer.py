import ray

from threading import Barrier


@ray.remote
class Synchronizer:
    def __init__(self, islands):
        self.barrier = Barrier(islands)

    def wait(self):
        self.barrier.wait(timeout=60)
