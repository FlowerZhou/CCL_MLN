"""
Markov Logic Network
"""
from typing import Iterator

from baize.logic import Predicate
from baize.utils.vocab import Vocabulary


class MarkovLogicNet(object):
    """
    Markov logic net
    """

    def __init__(self):

        self._schemas = dict()
        self._rules = []


    def add_predicate(self, predicate: Predicate):
        """
        add predicate to the schema system
        """
        self._schemas[predicate.name] = predicate

    def predicates(self) -> Iterator[Predicate]:
        """
        return the iterator for predicates
        """

        for pred in self._schemas.values():
            yield pred

    def get_predicate(self, name) -> Predicate:
        """
        get predicate with given name
        """

        return self._schemas[name]

    def add_rule(self, clause):
        """
        add rule to the system
        """

        self._rules.append(clause)

    def rules(self):
        """
        iterate through the rules
        """

        for rule in self._rules:
            yield rule

    def ground(self, evidences):
        from baize.mln.markov_random_field import MarkovRandomField

        return MarkovRandomField(self, evidences)


