import gc
import logging
from collections import defaultdict
from typing import List
import pytest
import matplotlib

from ether.core import Node

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from core.storage import StorageIndex
from core.utils import parse_size_string
from ether.vis import draw_basic
from ext.datalocality.characterization import get_function_characterizations
from ext.datalocality.cloudcpu import cloudcpu_settings
from ext.datalocality.etherdevices import convert_to_ether_nodes
from ext.datalocality.functionsim import DataLocalityHTTPSimulatorFactory
from ext.datalocality.generator import generate_devices
from ext.datalocality.oracles import DataLocalityFetOracle, DataLocalityResourceOracle
from ext.datalocality.resources import data_locality_resources_per_node_image, \
    data_locality_execution_time_distributions
from ext.datalocality.results import print_results, print_ml_f1, combined_figure, edge_figure, scheduling_tet, \
    save_schedule_results, skippy_tet_compbined_figure
from ext.datalocality.topology import urban_sensing_topology

from sim import docker
from sim.benchmark import Benchmark
from sim.core import Environment
from sim.docker import ImageProperties
from sim.faas import FunctionDeployment, FunctionRequest, Function, FunctionImage, ScalingConfiguration, \
    DeploymentRanking, FunctionContainer, KubernetesResourceConfiguration
from sim.faassim import Simulation
from sim.topology import Topology

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(format='%(asctime)s|%(levelname)-s|%(message)s', level=logging.DEBUG,
                        datefmt='%d.%m.%Y %H:%M:%S.%s')
    #for i in range(100, 5000, 500):
    #    logger.info('Executing simulation for %s nodes' % i)
    for scaling in range(1, 10):
        sim = simulation(1100, scaling)
        save_schedule_results(sim, 1100, scaling)
        del sim
        gc.collect()


#2600 351 451
#3600 51
#4100 51
#4600 51



def simulation(num_devices, scaling) -> Simulation:
    # logging.basicConfig(level=logging.DEBUG)

    # a topology holds the cluster configuration and network topology
    devices = generate_devices(num_devices)
    ether_nodes = convert_to_ether_nodes(devices)
    storage_index = StorageIndex(buckets=defaultdict(), tree=defaultdict(), items=defaultdict(),
                                 consume_node_item=defaultdict(), bucket_nodes=defaultdict())
    topology = urban_sensing_topology(ether_nodes, storage_index)

    # a benchmark is a simpy process that sets up the runtime system (e.g., creates container images, deploys functions)
    # and creates workload by simulating function requests
    benchmark = DataLocalityBenchmark(scaling)

    # a simulation runs until the benchmark process terminates\
    sim = Simulation(topology, benchmark)
    sim.env.storage_index = storage_index
    sim.run()
    #print_results(sim)
    return sim


# def data_locality_topology() -> Topology:
# t = Topology()
# scenario.UrbanSensingScenario().materialize(t)
# t.init_docker_registry()

#    num_devices = 0
#    devices = generate_devices(num_devices)
#    ether_nodes = convert_to_ether_nodes(devices)
#    storage_index = StorageIndex(buckets=defaultdict(), tree=defaultdict(), items=defaultdict())
#    topology = urban_sensing_topology(ether_nodes, storage_index)
#    plt.grid()
#    draw_basic(topology)
#    fig = plt.gcf()
#    fig.set_size_inches(100.5, 60.5)
#    plt.savefig('/Users/kenia/Desktop/test.png')
#    return topology


class DataLocalityBenchmark(Benchmark):
    scaling: int

    def __init__(self, scaling=1) -> None:
        self.scaling = scaling

    def setup(self, env: Environment):
        containers: docker.ContainerRegistry = env.container_registry
        fet_oracle = DataLocalityFetOracle(data_locality_execution_time_distributions)
        resource_oracle = DataLocalityResourceOracle(data_locality_resources_per_node_image)
        env.simulator_factory = DataLocalityHTTPSimulatorFactory(
            get_function_characterizations(resource_oracle, fet_oracle))

        # populate the global container registry with images
        containers.put(ImageProperties('keniack/f1-ml-pre', parse_size_string('282M'), arch='arm32'))
        containers.put(ImageProperties('keniack/f1-ml-pre', parse_size_string('282M'), arch='x86'))
        containers.put(ImageProperties('keniack/f1-ml-pre', parse_size_string('282M'), arch='aarch64'))

        containers.put(ImageProperties('keniack/f2-ml-train', parse_size_string('290M'), arch='arm32'))
        containers.put(ImageProperties('keniack/f2-ml-train', parse_size_string('290M'), arch='x86'))
        containers.put(ImageProperties('keniack/f2-ml-train', parse_size_string('290M'), arch='aarch64'))

        containers.put(ImageProperties('keniack/f3-ml-eval', parse_size_string('280M'), arch='arm32'))
        containers.put(ImageProperties('keniack/f3-ml-eval', parse_size_string('280M'), arch='x86'))
        containers.put(ImageProperties('keniack/f3-ml-eval', parse_size_string('280M'), arch='aarch64'))

        # log all the images in the container
        for name, tag_dict in containers.images.items():
            for tag, images in tag_dict.items():
                logger.info('%s, %s, %s', name, tag, images)

    def run(self, env: Environment):
        # deploy functions
        deployments = self.prepare_deployments()

        for deployment in deployments:
            yield from env.faas.deploy(deployment)

        # block until replicas become available (scheduling has finished and replicas have been deployed on the node)
        logger.info('waiting for replica')
        # yield env.process(env.faas.poll_available_replica('f1-ml-pre'))
        # yield env.process(env.faas.poll_available_replica('f2-ml-train'))
        yield env.process(env.faas.poll_available_replica('f3-ml-eval'))

        # run workload
        ps = []
        # execute 10 requests in parallel
        logger.info('executing 10 f1-ml-pre requests')
        # for i in range(100):
        #   ps.append(env.process(env.faas.invoke(FunctionRequest('f1-ml-pre'))))

        # execute 10 requests in parallel
        # for i in range(100):
        #   ps.append(env.process(env.faas.invoke(FunctionRequest('f2-ml-train'))))

        for i in range(1):
            ps.append(env.process(env.faas.invoke(FunctionRequest('f3-ml-eval'))))

        # wait for invocation processes to finish
        for p in ps:
            yield p

    def prepare_deployments(self) -> List[FunctionDeployment]:

        python_pi_fd = self.prepare_python_pi_deployment()

        python_f2_fd = self.prepare_ml_f2_train_deployment()

        python_f3_fd = self.prepare_ml_f3_eval_deployment()

        return [python_f3_fd]

    def prepare_ml_f3_eval_deployment(self):
        # Design Time

        python_pi = 'f3-ml-eval'
        python_pi_cpu = FunctionImage(image='keniack/f3-ml-eval')
        python_pi_fn = Function(python_pi, fn_images=[python_pi_cpu])

        # Run time
        data_storage = {
            'skippy.io.data.consume.1': 'train_bucket.model.npy',
        }
        labels = {'watchdog': 'http', 'workers': '1', 'cluster': '1'}
        python_pi_fn_container = FunctionContainer(python_pi_cpu)
        python_pi_fn_container.labels.update(data_storage)
        python_pi_fn_container.labels.update(labels)
        scaling_config = ScalingConfiguration()
        scaling_config.scale_min = self.scaling

        python_pi_fd = FunctionDeployment(
            python_pi_fn,
            [python_pi_fn_container],
            scaling_config
        )

        return python_pi_fd

    def prepare_python_pi_deployment(self):
        # Design Time

        python_pi = 'f1-ml-pre'
        python_pi_cpu = FunctionImage(image='keniack/f1-ml-pre')
        python_pi_fn = Function(python_pi, fn_images=[python_pi_cpu])

        # Run time
        data_storage = {
            'skippy.io.data.consume.1': 'pre_bucket.test_file.csv',
            'skippy.io.data.produce.1': 'train_bucket.treated_data.npy',
        }
        labels = {'watchdog': 'http', 'workers': '1', 'cluster': '1'}
        python_pi_fn_container = FunctionContainer(python_pi_cpu)
        python_pi_fn_container.labels.update(data_storage)
        python_pi_fn_container.labels.update(labels)

        python_pi_fd = FunctionDeployment(
            python_pi_fn,
            [python_pi_fn_container],
            ScalingConfiguration()
        )

        return python_pi_fd

    def prepare_ml_f2_train_deployment(self):
        # Design Time

        python_pi = 'f2-ml-train'
        python_pi_cpu = FunctionImage(image='keniack/f2-ml-train')
        python_pi_fn = Function(python_pi, fn_images=[python_pi_cpu])

        # Run time
        data_storage = {
            'skippy.io.data.consume.1': 'train_bucket.treated_data.npy',
            'skippy.io.data.produce.1': 'train_bucket.model.npy',
        }
        labels = {'watchdog': 'http', 'workers': '1', 'cluster': '1'}
        python_pi_fn_container = FunctionContainer(python_pi_cpu)
        python_pi_fn_container.labels.update(data_storage)
        python_pi_fn_container.labels.update(labels)

        python_pi_fd = FunctionDeployment(
            python_pi_fn,
            [python_pi_fn_container],
            ScalingConfiguration()
        )

        return python_pi_fd


def print_topology(topology):
    for n in topology.nodes:
        if type(n) is Node:
            logger.info('Node %s locality %s' % (n.name, n.labels.get('locality.skippy.io/type')))


if __name__ == '__main__':
    main()
