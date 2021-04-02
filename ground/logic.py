"""
grounded logic
"""
from enum import Enum
from typing import Union

from loguru import logger

from baize.logic import Atom, Predicate
from baize.utils.vocab import Vocabulary

predicate_voc = Vocabulary()
constant_voc = Vocabulary(index_base=10000)


class TruthValue(Enum):
    FALSE = 0
    TRUE = 1
    UNKNOWN = 2


class GroundedAtom(object):

    def __init__(self, x: Union[Atom, Predicate], const_args=None):

        if isinstance(x, Atom):
            predicate = x.predicate
        else:
            predicate = x
        self.close_word = predicate.closed_world
        self.predicate = predicate_voc.add(predicate.name)
        if const_args is None:
            self.terms = tuple()
        else:
            if isinstance(const_args[0], int):  # has been mapped into ID
                self.terms = tuple(const_args)
            else:
                self.terms = tuple(constant_voc.add(arg) for arg in const_args)

    def __eq__(self, other):
        return self.predicate == other.predicate and self.terms == other.terms

    def __hash__(self):
        return hash((self.predicate, self.terms))

    def __str__(self):
        return predicate_voc[self.predicate] + \
               "(" + ",".join(constant_voc[arg] for arg in self.terms) + ")"


class GroundedLiteral(object):

    def __init__(self, grounded_atom: GroundedAtom =None, sense: bool=True):

        self.atom = grounded_atom
        self.sense = sense

    def __str__(self):

        prefix = "!" if not self.sense else ""
        return prefix + " " + str(self.atom)

    def truth(self, evidence_truth) -> TruthValue:

        if self.atom in evidence_truth:
            evidence_value = evidence_truth[self.atom]
            if self.sense == evidence_value:
                return TruthValue.TRUE
            else:
                return TruthValue.FALSE
        else:
            if self.atom.close_word:
                return TruthValue.FALSE
            else:
                return TruthValue.UNKNOWN


class GroundedClause(object):

    def __init__(self, literals=None):

        if literals is None:
            self.literals = []
        else:
            self.literals = literals

    def __str__(self):

        return "v".join(str(literal) for literal in self.literals)

    @staticmethod
    def grounded_from(clause, assignment):
        """
        create from a clause with assignment to variables
        """
        grounded_literals = []
        for literal in clause.literals:
            grounded_terms = []

            for term in literal.atom.terms:

                if term.is_var:
                    ground_term = assignment[term.name]
                else:
                    ground_term = term
                grounded_terms.append(ground_term)

            grounded_atom = GroundedAtom(literal.atom, grounded_terms)
            grounded_literal = GroundedLiteral(grounded_atom, sense=literal.sense)
            grounded_literals.append(grounded_literal)

        return GroundedClause(literals=grounded_literals)

    def truth(self, evidence_truth):

        truth_values = [literal.truth(evidence_truth) for literal in self.literals]
        if any(x == TruthValue.TRUE for x in truth_values):
            return TruthValue.TRUE
        elif all(x == TruthValue.FALSE for x in truth_values):
            return TruthValue.FALSE
        else:
            return TruthValue.UNKNOWN



