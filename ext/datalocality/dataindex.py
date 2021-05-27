import logging
import random

from typing import List, Set

from ether.core import Node, Link
from sim.core import Environment

logger = logging.getLogger(__name__)


def get_best_node(bucket: str, file_name: str, node_name, env: Environment, file_size, upload=False):
    return predict_storage_node(bucket, file_name, node_name, env, file_size, upload)
    #return random_node(bucket, file_name, node_name, env, file_size, upload)


def predict_storage_node(bucket: str, file_name: str, node_name, env: Environment, file_size, upload=False):
    localities = ["edge", "cloud"]
    best_node = None
    for locality in localities:
        storage_nodes = env.storage_index.get_bucket_nodes(bucket) if upload else \
            env.storage_index.get_data_nodes(bucket, file_name)
        logger.info('Storage Nodes filtered at %s %s' % (locality, storage_nodes))
        best_node = calculate_best_storage_node(storage_nodes, env, node_name, file_size, bucket, file_name, locality)
        if best_node:
            break
    return best_node


def random_node(bucket: str, file_name: str, node_name, env: Environment, file_size, upload=False):
    storage_nodes = env.storage_index.get_bucket_nodes(bucket) if upload else \
        env.storage_index.get_data_nodes(bucket, file_name)
    logger.info('Storage Nodes filtered at %s ' % (storage_nodes))
    best_node = random.choice([s for s in storage_nodes if s != node_name])
    return best_node


def calculate_best_storage_node(storage_nodes: Set[str], env: Environment, node_name, file_size, bucket: str,
                                file_name: str, locality: str):
    time = 0
    max_bw = 0
    max_bw_storage = None
    logger.debug('Calculations node [%s] to storage options %s to transfer %s/%s' % (
        node_name, storage_nodes, bucket, file_name))
    for storage in storage_nodes:
        nodes = list(filter(lambda n: type(n) is Node and n.name == storage and
                                      n.labels['locality.skippy.io/type'] == locality, env.topology.nodes))
        if len(nodes) == 0:
            logger.debug('No storage node on %s' % locality)
            continue
        logger.debug('Node [%s] to storage [%s] on %s' % (node_name, storage, locality))
        if storage == node_name:
            logger.debug('Node and storage the same. Skipping')
            continue

        bandwidth = env.cluster.get_dl_bandwidth(storage, node_name)
        bandwidth = bandwidth * 125000 * 0.97 if bandwidth else None
        if bandwidth is not None and bandwidth > max_bw:
            max_bw = bandwidth
            max_bw_storage = storage

    if max_bw_storage and file_size:
        time += file_size / max_bw
    logger.debug('[%s] is the best storage from node [%s] to transfer %s/%s. Time = %.2f sec' % (
        max_bw_storage, node_name, bucket, file_name, time))
    return max_bw_storage
