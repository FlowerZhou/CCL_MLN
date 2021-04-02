"""
Markov random field
"""
import itertools
from collections import defaultdict

from baize.logic import Atom, Clause, Term, Literal
from baize.mln.markov_logic_network import MarkovLogicNet


class MarkovRandomField(object):

    def __init__(self, mln: MarkovLogicNet, evidences):

        self.mln = mln

        self.evidences = evidences

        self._find_domain_instances()

        self._find_all_ground_atoms()

    def _find_domain_instances(self):

        self.domain_instances = defaultdict(set)

        for evidence in self.evidences:
            pred = evidence.atom.predicate
            for domain, instance in zip(pred.args, evidence.atom.terms):
                self.domain_instances[domain].add(instance)

    def _find_all_ground_atoms(self):

        self.ground_atoms = []

        for pred in self.mln.predicates():
            pass






