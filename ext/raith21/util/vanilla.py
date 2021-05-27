from core.priorities import BalancedResourcePriority, ImageLocalityPriority
from ext.raith21.util import predicates


def get_predicates(fet_oracle, resource_oracle):
    return predicates.get_predicates(fet_oracle, resource_oracle)


def get_priorities():
    return [
        (1, BalancedResourcePriority()),
        (1, ImageLocalityPriority()),
    ]
