import logging
from collections import defaultdict
from typing import List
import pytest
import matplotlib

from ether.core import Node
from ext.datalocality.results import scheduling_scalability

matplotlib.use('TkAgg')


logger = logging.getLogger(__name__)


def main():
    scheduling_scalability()
    #skippy_tet_compbined_figure()
    #prediction_tet()


if __name__ == '__main__':
    main()
