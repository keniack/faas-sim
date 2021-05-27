from collections import Counter, defaultdict
from enum import Enum
from typing import List, Callable

import numpy as np

from .device import  Device
from .model import Arch, Accelerator, Bins, Location, Connection, Disk, Requirements


def count_attribute(devices: List[Device], values: List, getter: Callable[[Device], Enum]):
    counter = {}
    if len(devices) == 0:
        return {}
    for attr in values:
        counter[attr] = 0
    for device in devices:
        counter[getter(device)] += 1
    percentage = {}
    n_devices = len(devices)
    for attr, count in counter.items():
        percentage[attr] = count / n_devices
    return percentage


def calculate_requirements(devices: List[Device]) -> Requirements:
    arch = count_attribute(devices, list(Arch), lambda d: d.arch)
    accelerator = count_attribute(devices, list(Accelerator), lambda d: d.accelerator)
    cores = count_attribute(devices, list(Bins), lambda d: d.cores)
    location = count_attribute(devices, list(Location), lambda d: d.location)
    connection = count_attribute(devices, list(Connection), lambda d: d.connection)
    network = count_attribute(devices, list(Bins), lambda d: d.network)
    cpu_mhz = count_attribute(devices, list(Bins), lambda d: d.cpu_mhz)
    cpu = count_attribute(devices, list(set([x.cpu for x in devices])), lambda d: d.cpu)
    ram = count_attribute(devices, list(Bins), lambda d: d.ram)
    disk = count_attribute(devices, list(Disk), lambda d: d.disk)
    return Requirements(
        arch=arch,
        accelerator=accelerator,
        cores=cores,
        location=location,
        connection=connection,
        network=network,
        cpu_mhz=cpu_mhz,
        cpu=cpu,
        ram=ram,
        disk=disk,
        gpu_model=defaultdict(),
        gpu_vram=defaultdict(),
        gpu_mhz=defaultdict()
    )


def calculate_heterogeneity(p: Requirements, q: Requirements) -> float:
    entropy_p = 0
    entropy_q = 0
    for (p_enum, p_characteristic), (q_enum, q_characteristic) in zip(p.characteristics, q.characteristics):
        for value in list(p_enum):
            default_val = 0.0000000000000000000001
            p_char = p_characteristic.get(value, default_val)
            if p_char == 0:
                p_char = default_val
            entropy_p += p_char * np.log(p_char)
            q_char = q_characteristic.get(value, default_val)
            if q_char == 0:
                q_char = default_val
            entropy_q += q_char * np.log(q_char)

    return entropy_p - entropy_q
