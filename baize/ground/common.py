"""
common utilities
"""

import itertools

from baize.ground.logic import GroundedAtom


def ground_predicate(predicate, domain_instances):
    """
    grounding the predicate
    """
    arg_instances = [domain_instances[arg.type] for arg in predicate.args]
    for params in itertools.product(*arg_instances):
        terms = list(params)
        yield GroundedAtom(predicate, terms)

