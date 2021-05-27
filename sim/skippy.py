"""
Module that glues simulation concepts to skippy concepts.
"""
import copy
import logging
import operator
import random
from collections import defaultdict
from functools import reduce
from typing import List, Dict

from ether.core import Node as EtherNode

from core.clustercontext import ClusterContext
from core.model import ImageState, PodSpec, Container, Pod, ResourceRequirements, Node as SkippyNode, \
    Capacity as SkippyCapacity
from core.storage import StorageIndex
from core.utils import counter
from sim import docker
from sim.core import Environment
from sim.faas import FunctionContainer, FunctionDeployment
from sim.topology import LazyBandwidthGraph, DockerRegistry

logger = logging.getLogger(__name__)


class SimulationClusterContext(ClusterContext):

    def __init__(self, env: Environment):
        self.env = env

        self.topology = env.topology
        self.container_registry: docker.ContainerRegistry = env.container_registry
        self.bw_graph = None
        self.nodes = None

        super().__init__()

        self.storage_index = env.storage_index or StorageIndex()
        self._storage_nodes = None
        #self.create_node_item_index()


    def create_node_item_index(self):
        for node in self.list_nodes():
            for bucket, file_name in self.storage_index.items:
                storage_nodes = self.storage_index.get_data_nodes(bucket, file_name)
                storage_nodes = [s for s in storage_nodes if s != node.name]
                bw = self.get_bandwidth_graph()[node.name]
                fastest_storage_node = max(storage_nodes, key=lambda n: bw[n])
                k = (node.name, bucket, file_name)
                self.storage_index.consume_node_item[k] = fastest_storage_node
            for bucket, storage_nodes in self.storage_index.buckets.items():
                storage_nodes = [s for s in storage_nodes if s != node.name]
                bw = self.get_bandwidth_graph()[node.name]
                fastest_storage_node = max(storage_nodes, key=lambda n: bw[n])
                k = (node.name, bucket)
                self.storage_index.bucket_nodes[k] = fastest_storage_node

    def get_init_image_states(self) -> Dict[str, ImageState]:
        # FIXME: fix this image state business in skippy
        return defaultdict(lambda: None)

    def retrieve_image_state(self, image_name: str) -> ImageState:
        # FIXME: hacky workaround
        images = self.container_registry.find(image_name)

        if not images:
            raise ValueError('No container image "%s"' % image_name)

        if len(images) == 1 and images[0].arch is None:
            sizes = {
                'x86': images[0].size,
                'arm': images[0].size,
                'arm32': images[0].size,
                'arm32v7': images[0].size,
                'aarch64': images[0].size,
                'arm64': images[0].size,
                'amd64': images[0].size
            }
        else:
            sizes = {image.arch: image.size for image in images if image.arch is not None}

        return ImageState(sizes)

    def get_bandwidth_graph(self):
        if self.bw_graph is None:
            self.bw_graph = LazyBandwidthGraph(self.topology)

        return self.bw_graph

    def list_nodes(self) -> List[SkippyNode]:
        if self.nodes is None:
            self.nodes = [to_skippy_node(node) for node in self.topology.get_nodes() if node != DockerRegistry]

        return self.nodes

    def get_next_storage_node(self, node: SkippyNode) -> str:
        if self.is_storage_node(node):
            return node.name
        if not self.storage_nodes:
            return None

        bw = self.get_bandwidth_graph()[node.name]
        storage_nodes = list(self.storage_nodes.values())
        random.shuffle(storage_nodes)  # make sure you get a random one if bandwidth is the same
        storage_node = max(storage_nodes, key=lambda n: bw[n.name])

        return storage_node.name

    def get_storage_nodes(self, bucket: str, name: str) -> List[str]:
        storage_list = self.storage_index.get_bucket_nodes(bucket)
        if storage_list is None:
            return None
        return list(storage_list)

    def get_storage_index(self) -> StorageIndex:
        items = defaultdict(dict)
        buckets = defaultdict(set)
        tree = defaultdict(lambda: defaultdict(list))
        return StorageIndex(buckets, tree, items)

    @property
    def storage_nodes(self) -> Dict[str, SkippyNode]:
        if self._storage_nodes:
            return self._storage_nodes

        s_nodes = reduce(operator.or_, self.storage_index.get_storage_nodes())
        if s_nodes:
            self._storage_nodes = {node.name: node for node in self.list_nodes() if node.name in s_nodes}

        return self._storage_nodes

    def is_storage_node(self, node: SkippyNode):
        return 'data.skippy.io/storage' in node.labels


def to_skippy_node(node: EtherNode) -> SkippyNode:
    """
    Converts an ether Node into a skippy model Node.
    :param node: the node to convert
    :return: the skippy node
    """
    capacity = SkippyCapacity(node.capacity.cpu_millis, node.capacity.memory)
    allocatable = copy.copy(capacity)

    labels = dict(node.labels)
    labels['beta.kubernetes.io/arch'] = node.arch

    return SkippyNode(node.name, capacity=capacity, allocatable=allocatable, labels=labels)


pod_counters = defaultdict(counter)


def create_function_pod(fd: 'FunctionDeployment', fn: 'FunctionContainer') -> Pod:
    """
    Creates a new Pod that hosts the given function.
    :param fd: the function deployment to get the deployed function name
    :param fn: the function container to package
    :return: the Pod
    """
    requests = fn.resource_config.get_resource_requirements()
    resource_requirements = ResourceRequirements(requests)

    spec = PodSpec()
    spec.containers = [Container(fn.image, resource_requirements)]
    spec.labels = fn.labels

    cnt = next(pod_counters[fd.name])
    pod = Pod(f'pod-{fd.name}-{cnt}', 'faas-sim')
    pod.spec = spec

    return pod
