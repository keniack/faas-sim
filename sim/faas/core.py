import abc
import enum
import logging
from collections import defaultdict
from typing import List, Dict, NamedTuple, Optional

from ether.util import parse_size_string
from skippy.core.model import Pod

from sim.core import Environment, NodeState
from sim.oracle.oracle import FetOracle, ResourceOracle

logger = logging.getLogger(__name__)


def counter(start: int = 1):
    n = start
    while True:
        yield n
        n += 1


class FunctionState(enum.Enum):
    CONCEIVED = 1
    STARTING = 2
    RUNNING = 3
    SUSPENDED = 4


class Resources:
    memory: int
    cpu: int

    def __init__(self, cpu_millis: int = 1 * 1000, memory: int = 1 * 1024 * 1024):
        self.memory = memory
        self.cpu = cpu_millis

    def __str__(self):
        return 'Resources(CPU: {0} Memory: {1})'.format(self.cpu, self.memory)

    @staticmethod
    def from_str(memory, cpu):
        """
        :param memory: "64Mi"
        :param cpu: "250m"
        :return:
        """
        return Resources(int(cpu.rstrip('m')), parse_size_string(memory))


class FunctionResourceCharacterization:
    cpu: float
    blkio: float
    gpu: float
    net: float
    ram: float

    def __init__(self, cpu: float, blkio: float, gpu: float, net: float, ram: float):
        self.cpu = cpu
        self.blkio = blkio
        self.gpu = gpu
        self.net = net
        self.ram = ram

    def __len__(self):
        return 5

    def __delitem__(self, key):
        self.__delattr__(key)

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)


class FunctionCharacterization:

    def __init__(self, image: str, fet_oracle: FetOracle, resource_oracle: ResourceOracle):
        self.image = image
        self.fet_oracle = fet_oracle
        self.resource_oracle = resource_oracle

    def sample_fet(self, host: str) -> Optional[float]:
        return self.fet_oracle.sample(host, self.image)

    def get_resources_for_node(self, host: str) -> FunctionResourceCharacterization:
        return self.resource_oracle.get_resources(host, self.image)


class DeploymentRanking:
    # TODO probably better to remove default/enable default for one image
    images: List[str] = ['tpu', 'gpu', 'cpu']

    def __init__(self, images: List[str]):
        self.images = images

    def set_first(self, image: str):
        index = self.images.index(image)
        updated = self.images[:index] + self.images[index + 1:]
        self.images = [image] + updated

    def get_first(self):
        return self.images[0]


class FunctionDefinition:
    # TODO is this useful on this level? can we say something about the other architectures?
    # or would it be too cumbersome to provide this for each image?
    requests: Resources = Resources()

    name: str

    # the manifest list name
    image: str

    # characterization per image
    characterization: FunctionCharacterization

    labels: Dict[str, str]

    def __init__(self, name: str, image: str, characterization: FunctionCharacterization = None,
                 labels: Dict[str, str] = None):
        self.name = name
        self.image = image
        self.characterization = characterization
        if labels is None:
            self.labels = {}
        else:
            self.labels = labels

    def get_resource_requirements(self) -> Dict:
        return {
            'cpu': self.requests.cpu,
            'memory': self.requests.memory
        }

    def sample_fet(self, host: str) -> Optional[float]:
        return self.characterization.sample_fet(host)

    def get_resources_for_node(self, host: str):
        return self.characterization.get_resources_for_node(host)


class FunctionDeployment:
    name: str
    function_definitions: Dict[str, FunctionDefinition]

    # used to determine which function to take when scaling
    ranking: DeploymentRanking

    scale_min: int = 1
    scale_max: int = 20
    scale_factor: int = 1
    scale_zero: bool = False

    # percentages of scaling per image, can be used to hinder scheduler to overuse expensive resources (i.e. tpu)
    function_factor: Dict[str, float]

    # average requests per second threshold for scaling
    rps_threshold: int = 20

    # window over which to track the average rps
    alert_window: int = 50  # TODO currently not supported by FaasRequestScaler

    # seconds the rps threshold must be violated to trigger scale up
    rps_threshold_duration: int = 10

    # target average cpu utilization of all replicas, used by HPA
    target_average_utilization: float = 0.5

    # target average rps over all replicas, used by AverageFaasRequestScaler
    target_average_rps: int = 200

    # target of maximum requests in queue
    target_queue_length: int = 75

    target_average_rps_threshold = 0.1

    def __init__(self, name: str, function_definitions: Dict[str, FunctionDefinition],
                 ranking: DeploymentRanking = None, function_factor=None):
        self.name = name
        self.image = name  #
        self.function_definitions = function_definitions
        if ranking is None:
            self.ranking = DeploymentRanking(list(function_definitions.keys()))
        else:
            self.ranking = ranking
        if function_factor is None:
            function_factor = {}
            for image in function_definitions.keys():
                function_factor[image] = 1

        self.function_factor = function_factor

    def get_selected_service(self):
        return self.function_definitions[self.ranking.get_first()]

    def get_services(self):
        return list(map(lambda i: self.function_definitions[i], self.ranking.images))


class FunctionReplica:
    """
    A function replica is an instance of a function running on a specific node.
    """
    function: FunctionDefinition
    node: NodeState
    pod: Pod
    state: FunctionState = FunctionState.CONCEIVED

    simulator: 'FunctionSimulator' = None


class FunctionRequest:
    request_id: int
    name: str
    size: float = None

    id_generator = counter()

    def __init__(self, name, size=None) -> None:
        super().__init__()
        self.name = name
        self.size = size
        self.request_id = next(self.id_generator)

    def __str__(self) -> str:
        return 'FunctionRequest(%d, %s, %s)' % (self.request_id, self.name, self.size)

    def __repr__(self):
        return self.__str__()


class FunctionResponse(NamedTuple):
    request_id: int
    code: int
    t_wait: float = 0
    t_exec: float = 0
    node: str = None


class FaasSystem(abc.ABC):

    @abc.abstractmethod
    def deploy(self, fn: FunctionDeployment): ...

    @abc.abstractmethod
    def invoke(self, request: FunctionRequest): ...

    @abc.abstractmethod
    def remove(self, fn: FunctionDeployment): ...

    @abc.abstractmethod
    def get_deployments(self) -> List[FunctionDeployment]: ...

    @abc.abstractmethod
    def get_function_index(self) -> Dict[str, FunctionDefinition]: ...

    @abc.abstractmethod
    def get_replicas(self, fn_name: str, state=None) -> List[FunctionReplica]: ...

    @abc.abstractmethod
    def scale_down(self, function_name: str, remove: int): ...

    @abc.abstractmethod
    def scale_up(self, function_name: str, replicas: int): ...

    @abc.abstractmethod
    def discover(self, function: FunctionDefinition) -> List[FunctionReplica]: ...

    @abc.abstractmethod
    def suspend(self, function_name: str): ...


class LoadBalancer:
    env: Environment
    replicas: Dict[str, List[FunctionReplica]]

    def __init__(self, env, replicas) -> None:
        super().__init__()
        self.env = env
        self.replicas = replicas

    def get_running_replicas(self, function: str):
        return [replica for replica in self.replicas[function] if replica.state == FunctionState.RUNNING]

    def next_replica(self, request: FunctionRequest) -> FunctionReplica:
        raise NotImplementedError


class RoundRobinLoadBalancer(LoadBalancer):

    def __init__(self, env, replicas) -> None:
        super().__init__(env, replicas)
        self.counters = defaultdict(lambda: 0)

    def next_replica(self, request: FunctionRequest) -> FunctionReplica:
        replicas = self.get_running_replicas(request.name)
        i = self.counters[request.name] % len(replicas)
        self.counters[request.name] = (i + 1) % len(replicas)

        replica = replicas[i]

        return replica


class FunctionSimulator(abc.ABC):

    def deploy(self, env: Environment, replica: FunctionReplica):
        yield env.timeout(0)

    def startup(self, env: Environment, replica: FunctionReplica):
        yield env.timeout(0)

    def setup(self, env: Environment, replica: FunctionReplica):
        yield env.timeout(0)

    def invoke(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
        yield env.timeout(0)

    def teardown(self, env: Environment, replica: FunctionReplica):
        yield env.timeout(0)


class SimulatorFactory:

    def create(self, env: Environment, fn: FunctionDefinition) -> FunctionSimulator:
        raise NotImplementedError
