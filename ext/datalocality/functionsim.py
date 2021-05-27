import logging
from random import randrange
from typing import Callable, Optional, Dict

from simpy import Resource

from sim.core import Environment
from sim.docker import pull as docker_pull
from sim.faas import FunctionSimulator, FunctionRequest, FunctionReplica, SimulatorFactory, simulate_data_download, \
    simulate_data_upload, FunctionCharacterization, FunctionContainer


def linear_queue_fet_increase(current_requests: int, max_requests: int) -> float:
    return current_requests / max_requests


logger = logging.getLogger(__name__)


class PythonHTTPSimulator(FunctionSimulator):

    def __init__(self, queue: Resource, scale: Callable[[int, int], float], fn: FunctionContainer,
                 characterization: FunctionCharacterization):
        self.worker_threads = queue.capacity
        self.queue = queue
        self.scale = scale
        self.delay = 0
        self.fn = fn
        self.characterization = characterization

    def invoke(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
        token = self.queue.request()
        yield token  # wait for access

        # because of GIL and Threads, we can easily estimate the additional time caused by concurrent requests to the
        # same Function
        factor = max(1, self.scale(self.queue.count, self.queue.capacity))
        try:
            fet = self.characterization.sample_fet(replica.node.name)
            if fet is None:
                logging.error(f"FET for node {replica.node.name} for function {self.fn.image} was not found")
                raise ValueError(f'{replica.node.name}')
            fet = float(fet) * factor
            yield env.timeout(fet)


        except KeyError:
            pass

        self.queue.release(token)


class PythonHttpSimulatorFactory(SimulatorFactory):

    def __init__(self, fn_characterizations: Dict[str, FunctionCharacterization]):
        self.fn_characterizations = fn_characterizations

    def create(self, env: Environment, fn: FunctionContainer) -> FunctionSimulator:
        workers = int(fn.labels['workers'])
        queue = Resource(env=env, capacity=workers)
        return PythonHTTPSimulator(queue, linear_queue_fet_increase, fn, self.fn_characterizations[fn.image])


class FunctionCall:
    replica: FunctionReplica
    request: FunctionRequest
    start: int
    end: Optional[int] = None

    def __init__(self, request, replica, start, end=None):
        self.request = request
        self.replica = replica
        self.start = start
        self.end = end

    @property
    def request_id(self):
        return self.request.request_id


class InterferenceAwarePythonHttpSimulatorFactory(SimulatorFactory):

    def __init__(self, fn_characterizations: Dict[str, FunctionCharacterization]):
        self.fn_characterizations = fn_characterizations

    def create(self, env: Environment, fn: FunctionContainer) -> FunctionSimulator:
        workers = int(fn.labels['workers'])
        queue = Resource(env=env, capacity=workers)
        return InterferenceAwarePythonHttpSimulator(queue, linear_queue_fet_increase, fn,
                                                    self.fn_characterizations[fn.image])


class DataLocalityHTTPSimulatorFactory(SimulatorFactory):

    def __init__(self, fn_characterizations: Dict[str, FunctionCharacterization]):
        self.fn_characterizations = fn_characterizations

    def create(self, env: Environment, fn: FunctionContainer) -> FunctionSimulator:
        workers = 0
        if fn.labels.get('workers'):
            workers = int(fn.labels['workers'])
        queue = Resource(env=env, capacity=workers)
        return DataLocalityHTTPSimulator(queue, linear_queue_fet_increase, fn, self.fn_characterizations[fn.image])


class DataLocalityHTTPSimulator(FunctionSimulator):
    def __init__(self, queue: Resource, scale: Callable[[int, int], float], fn: FunctionContainer,
                 characterization: FunctionCharacterization):
        self.worker_threads = queue.capacity
        self.queue = queue
        self.scale = scale
        self.deployment = fn
        self.delay = 0
        self.characterization = characterization

    def deploy(self, env: Environment, replica: FunctionReplica):
        yield from docker_pull(env, replica.image, replica.node.ether_node)

    def setup(self, env: Environment, replica: FunctionReplica):
        image = replica.pod.spec.containers[0].image
        if 'inference' in image:
            yield from simulate_data_download(env, replica)

    def invoke(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
        token = self.queue.request()
        t_wait_start = env.now
        yield token  # wait for access
        t_wait_end = env.now
        t_fet_start = env.now
        # because of GIL and Threads, we can easily estimate the additional time caused by concurrent requests to the
        # same Function
        factor = max(1, self.scale(self.queue.count, self.queue.capacity))
        try:
            fet = self.characterization.sample_fet(replica.node.name)
            if fet is None:
                logging.error(f"FET for node {replica.node.name} for function {self.deployment.image} was not found")
                raise ValueError(f'{replica.node.name}')
            fet = float(fet) * factor

            image = replica.pod.spec.containers[0].image
            if 'ml-pre' in image or 'train' or 'eval' in image:
                yield from simulate_data_download(env, replica)
            start = env.now
            call = FunctionCall(request, replica, start)
            replica.node.all_requests.append(call)
            yield env.timeout(fet)
            if 'ml-pre' in image or 'train' or 'eval' in image:
                yield from simulate_data_upload(env, replica)
            t_fet_end = env.now
            locality = env.topology.find_node(replica.node.name).labels.get('locality.skippy.io/type') \
                if env.topology.find_node(replica.node.name) else None
            #t_duration = (t_fet_end - t_fet_start) + randrange(250, 300)
            t_duration = (t_fet_end - t_fet_start)
            env.metrics.log_fet(request.name, replica.image, replica.node.name, t_fet_start, t_fet_end, 1, id(replica),
                                **{'t_wait_start': t_wait_start, 't_wait_end': t_wait_end,
                                   't_tet': t_duration, 't_node_locality': locality})
            replica.node.set_end(request.request_id, t_fet_end)
        except KeyError:
            pass

        self.queue.release(token)


class InterferenceAwarePythonHttpSimulator(FunctionSimulator):
    def __init__(self, queue: Resource, scale: Callable[[int, int], float], fn: FunctionContainer,
                 characterization: FunctionCharacterization):
        self.worker_threads = queue.capacity
        self.queue = queue
        self.scale = scale
        self.deployment = fn
        self.delay = 0
        self.characterization = characterization

    def deploy(self, env: Environment, replica: FunctionReplica):
        yield from docker_pull(env, replica.image, replica.node.ether_node)

    def setup(self, env: Environment, replica: FunctionReplica):
        image = replica.pod.spec.containers[0].image
        if 'inference' in image:
            yield from simulate_data_download(env, replica)

    def invoke(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
        token = self.queue.request()
        t_wait_start = env.now
        yield token  # wait for access
        t_wait_end = env.now
        t_fet_start = env.now
        # because of GIL and Threads, we can easily estimate the additional time caused by concurrent requests to the
        # same Function
        factor = max(1, self.scale(self.queue.count, self.queue.capacity))
        try:
            fet = self.characterization.sample_fet(replica.node.name)
            if fet is None:
                logging.error(f"FET for node {replica.node.name} for function {self.deployment.image} was not found")
                raise ValueError(f'{replica.node.name}')
            fet = float(fet) * factor

            image = replica.pod.spec.containers[0].image
            if 'preprocessing' in image or 'training' in image:
                yield from simulate_data_download(env, replica)
            start = env.now
            call = FunctionCall(request, replica, start)
            replica.node.all_requests.append(call)
            yield env.timeout(fet)

            # add degradation
            end = env.now
            degradation = replica.node.estimate_degradation(self.characterization.resource_oracle, start, end)
            delay = max(0, (fet * degradation) - fet)
            yield env.timeout(delay)
            if 'preprocessing' in image or 'training' in image:
                yield from simulate_data_upload(env, replica)
            t_fet_end = env.now
            env.metrics.log_fet(request.name, replica.image, replica.node.name, t_fet_start, t_fet_end,
                                t_wait_start, t_wait_end, degradation,
                                id(replica))
            replica.node.set_end(request.request_id, t_fet_end)
        except KeyError:
            pass

        self.queue.release(token)
