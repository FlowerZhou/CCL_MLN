"""
logic system
"""
from typing import List


class PredicateArgument(object):

    def __init__(self, type=None, name=None, unique=False):
        self.type = type
        self.name = name
        self.unique = unique


class Predicate(object):
    """
    Predicate
    """

    def __init__(self, name=None, args=None, closed_world=False):
        self.name = name
        self.args = args
        self.closed_world = closed_world


class Term(object):
    """
    A term in first-order logic; either a variable or a constant.
    """

    def __init__(self, is_var=False, name=None):

        self.is_var = is_var
        self.name = name


class Atom(object):
    """
    Atom formula: predicate on terms
    """
    def __init__(self, predicate=None, terms=None):

        self.predicate = predicate
        self.terms = terms


class Literal(object):
    """
    Literal: A literal in first-order logic.
    """

    def __init__(self, atom=None, sense=True):

        self.atom = atom
        self.sense = sense

    def flip_sense(self):
        """
        flip the sense, used in converting implication into conjunction
        """

        self.sense = not self.sense


class Clause(object):
    """
    A first-order logic clause, namely a disjunct of literals.
    """

    def __init__(self, weight=None, weight_fixed=False, literals=None, existantial_vars=None):

        self.weight = weight
        self.weight_fixed = weight_fixed
        if literals is None:
            self.literals = []
        else:
            self.literals = literals
        if existantial_vars is None:
            self.existential_vars = []
        else:
            self.existential_vars = existantial_vars


class Function(object):
    """
    Bool, numberic, and string functions; user-defined functions.
    """

    def __init__(self, name=None, args=None, type=None, builtin=False):
        self.name = name
        self.args = args
        self.type = type
        self.builtin = builtin


class Expression(object):
    """
    An expression to a function is like a literal to a predicate.
    The interesting part is that expressions can be nested.
    The value of an expression can be numeric/string/boolean.
    """

    def __init__(self):

        self.func: Function = None
        self.args: List[Expression] = []


class Evidence(object):

    def __init__(self):
        self.prior = None
        self.truth = None
        self.atom = None

