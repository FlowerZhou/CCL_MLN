"""
Variables
"""
from dnutils import ifnone

from mln.util import Interval


class MRFVariable(object):
    """
    Represents a (mutually exclusive) block of ground atoms.

    This is the base class for different types of variables an MRF
    may consist of, e.g. mutually exclusive ground atoms. The purpose
    of these variables is to provide some convenience methods for
    easy iteration over their values ("possible worlds") and to ease
    introduction of new types of variables in an MRF.

    The values of a variable should have a fixed order, so every value
    must have a fixed index.
    """

    def __init__(self, mrf, name, predicate, *ground_atoms):
        """
        :param mrf:         the instance of the MRF that this variable is added to
        :param name:        the readable name of the variable
        :param predicate:   the :class:`mln.base.Predicate` instance of this variable
        :param ground_atoms:    the ground atoms constituting this variable
        """
        self.mrf = mrf
        self.ground_atoms = list(ground_atoms)
        self.index = len(mrf.variables)
        self.name = name
        self.predicate = predicate

    def atom_values(self, value):
        """
        Returns a generator of (atom, value) pairs for the given variable value
        :param value:     a tuple of truth values
        """
        for atom, val in zip(self.ground_atoms, value):
            yield atom, val

    def iter_atoms(self):
        """
        Yields all ground atoms in this variable, sorted by atom index ascending
        """
        for atom in sorted(self.ground_atoms, key=lambda a: a.index):
            yield atom

    def str_val(self, value):
        """
        Returns a readable string representation for the value tuple given by `value`.
        """
        return '<%s>' % ', '.join(
            ['%s' % str(a_v[0]) if a_v[1] == 1 else ('!%s' % str(a_v[0]) if a_v[1] == 0 else '?%s?' % str(a_v[0]))
             for a_v in zip(self.ground_atoms, value)])

    def value_count(self, evidence=None):
        """
        Returns the number of values this variable can take.
        """
        raise Exception('%s does not implement value_count()' % self.__class__.__name__)

    def _iter_values(self, evidence=None):
        """
        Generates all values of this variable as tuples of truth values.
        :param evidence: an optional dictionary mapping ground atoms to truth values.

        """
        raise Exception('%s does not implement _iter_values()' % self.__class__.__name__)

    def value_index(self, value):
        """
        Computes the index of the given value.
        """
        raise Exception('%s does not implement value_index()' % self.__class__.__name__)

    def evidence_value_index(self, evidence=None):
        """
        Returns the index of this atomic block value for the possible world given in `evidence`.
        """
        value = self.evidence_value(evidence)
        if any(map(lambda v: v is None, value)):
            return None
        return self.value_index(tuple(value))

    def evidence_value(self, evidence=None):
        """
        Returns the value of this variable as a tuple of truth values
        in the possible world given by `evidence`.

        Exp: (0, 1, 0) for a mutex variable containing 3 ground atoms

        :param evidence:   the truth values wrt. the ground atom indices. Can be a
                           complete assignment of truth values (i.e. a list) or a dict
                           mapping ground atom indices to their truth values. If evidence is `None`,
                           the evidence vector of the MRF is taken.
        """
        if evidence is None:
            evidence = self.mrf.evidence
        value = []
        for ground_atom in self.ground_atoms:
            value.append(evidence[ground_atom.index])

        return tuple(value)

    def value2dict(self, value):
        """
        Takes a tuple of truth values and transforms it into a dict
        mapping the respective ground atom indices to their truth values.

        :param value: the value tuple to be converted.
        """
        evidence = {}
        for atom, val in zip(self.ground_atoms, value):
            evidence[atom.index] = val
        return evidence

    def setval(self, value, world):
        """
        Sets the value of this variable in the world `world` to the given value.

        :param value:    tuple representing the value of the variable.
        :param world:    vector representing the world to be modified:
        :returns:        the modified world.
        """
        for i, v in self.value2dict(value).items():
            world[i] = v
        return world

    def iter_values(self, evidence=None):
        """
        Iterates over (idx, value) pairs for this variable.

        Values are given as tuples of truth values of the respective ground atoms.
        For a binary variable (a 'normal' ground atom), for example, the two values
        are represented by (0,) and (1,). If `evidence is` given, only values
        matching the evidence values are generated.

        :param evidence: an optional dictionary mapping ground atom indices to truth values.

        .. warning:: ground atom indices are with respect to the mrf instance,
                                          not to the index of the ground atom in the variable

        .. warning:: The values are not necessarily order with respect to their
                     actual index obtained by `MRFVariable.value_index()`.

        """
        if type(evidence) is list:
            evidence = dict([(i, v) for i, v in enumerate(evidence)])
        for tup in self._iter_values(evidence):
            yield self.value_index(tup), tup

    def values(self, evidence=None):
        """
        Returns a generator of possible values of this variable under consideration of
        the evidence given, if any.

        Same as ``iter_values()`` but without value indices.
        """
        for _, val in self.iter_values(evidence):
            yield val

    def iter_worlds(self, evidence=None):
        """
        Iterates over possible worlds of evidence which can be generated with this variable.

        This does not have side effects on the `evidence`. If no `evidence` is specified,
        the evidence vector of the MRF is taken.

        :param evidence:     a possible world of truth values of all ground atoms in the MRF.
        :returns:
        """
        if type(evidence) is not dict:
            raise Exception('evidence must be of type dict, is %s' % type(evidence))
        if evidence is None:
            evidence = self.mrf.evidence_dicti()
        for i, val in self.iter_values(evidence):
            world = dict(evidence)
            value = self.value2dict(val)
            world.update(value)
            yield i, world

    def consistent(self, world, strict=False):
        """
        Checks for this variable if its assignment in the assignment `evidence` is consistent.

        :param world: the assignment to be checked.
        :param strict:   if True, no unknown assignments are allowed, i.e. there must not be any
                         ground atoms in the variable that do not have a truth value assigned.
        """
        total = 0
        evstr = ','.join([ifnone(world[atom.index], '?', str) for atom in self.ground_atoms])
        for ground_atom in self.ground_atoms:
            val = world[ground_atom.index]
            total += ifnone(val, 0)

        return True

    def __str__(self):
        return self.name

    def __repr__(self):
        return '<%s "%s": [%s]>' % (self.__class__.__name__, self.name, ','.join(map(str, self.ground_atoms)))

    def __contains__(self, element):
        return element in self.ground_atoms


class FuzzyVariable(MRFVariable):

    """
    Represents a fuzzy ground atom that can take values of truth in [0,1].

    It does not support iteration over values or value indexing.
    """

    def consistent(self, world, strict=False):
        value = self.evidence_value(world)[0]
        if value is not None:
            if 0 <= value <= 1:
                return True
        else:
            return True

    def value_count(self, evidence=None):
        if evidence is None or evidence[self.ground_atoms[0].index] is None:
            raise Exception('Cannot count number of values of an unassigned FuzzyVariable: %s' % str(self))
        else:
            return 1

    def iter_values(self, evidence=None):
        if evidence is None or evidence[self.ground_atoms[0].index] is None:
            raise Exception('Cannot iterate over values of fuzzy variables: %s' % str(self))
        else:
            yield None, (evidence[self.ground_atoms[0].index],)


class BinaryVariable(MRFVariable):
    """
    Represents a binary ("normal") ground atom with the two truth values 1 (true) and 0 (false).
    The first value is always the false one.
    """

    def value_count(self, evidence=None):
        if evidence is None:
            return 2
        else:
            return len(list(self.iter_values(evidence)))

    def _iter_values(self, evidence=None):
        if evidence is None:
            evidence = {}
        if len(self.ground_atoms) != 1:
            raise Exception('Illegal number of ground atoms in the variable %s' % repr(self))
        ground_atom = self.ground_atoms[0]
        if evidence.get(ground_atom.index) is not None and evidence.get(ground_atom.index) in (0, 1):
            yield (evidence[ground_atom.index],)
            return
        for t in (0, 1):
            yield (t,)

    def value_index(self, value):
        if value == (0,):
            return 0
        elif value == (1,):
            return 1
        else:
            raise Exception('Invalid world value for binary variable %s: %s' % (str(self), str(value)))

    def consistent(self, world, strict=False):
        val = world[self.ground_atoms[0].index]
        if strict and val is None:
            raise Exception('Invalid value of variable %s: %s' % (repr(self), val))


class MutexVariable(MRFVariable):
    """
    Represents a mutually exclusive block of ground atoms, i.e. a block
    in which exactly one ground atom must be true.
    """

    def value_count(self, evidence=None):
        if evidence is None:
            return len(self.ground_atoms)
        else:
            return len(list(self.iter_values(evidence)))

    def _iter_values(self, evidence=None):
        if evidence is None:
            evidence = {}
        atomindices = map(lambda a: a.index, self.ground_atoms)
        valpattern = []
        for mutexatom in atomindices:
            valpattern.append(evidence.get(mutexatom, None))
        # at this point, we have generated a value pattern with all values
        # that are fixed by the evidence argument and None for all others
        trues = sum(filter(lambda x: x == 1, valpattern))
        if trues > 1:  # sanity check
            raise Exception("More than one ground atom in mutex variable is true: %s" % str(self))
        if trues == 1:  # if the true value of the mutex var is in the evidence, we have only one possibility
            yield tuple(map(lambda x: 1 if x == 1 else 0, valpattern))
            return
        if all([x == 0 for x in valpattern]):
            raise Exception('Illegal value for a MutexVariable %s: %s' % (self, valpattern))
        for i, val in enumerate(valpattern):
            # generate a value tuple with a truth value for each atom which is not set to false by evidence
            if val == 0:
                continue
            elif val is None:
                values = [0] * len(valpattern)
                values[i] = 1
                yield tuple(values)

    def value_index(self, value):
        if sum(value) != 1:
            raise Exception('Invalid world value for mutex variable %s: %s' % (str(self), str(value)))
        else:
            return value.index(1)


class SoftMutexVariable(MRFVariable):
    """
    Represents a soft mutex block of ground atoms, i.e. a mutex block in which maximally
    one ground atom may be true.
    """

    def value_count(self, evidence=None):
        if evidence is None:
            return len(self.ground_atoms) + 1
        else:
            return len(list(self.iter_values(evidence)))

    def _iter_values(self, evidence=None):
        if evidence is None:
            evidence = {}
        atomindices = map(lambda a: a.index, self.ground_atoms)
        valpattern = []
        for mutexatom in atomindices:
            valpattern.append(evidence.get(mutexatom, None))
        # at this point, we have generated a value pattern with all values
        # that are fixed by the evidence argument and None for all others
        trues = sum(filter(lambda x: x == 1, valpattern))
        if trues > 1:  # sanity check
            raise Exception("More than one ground atom in mutex variable is true: %s" % str(self))
        if trues == 1:  # if the true value of the mutex var is in the evidence, we have only one possibility
            yield tuple(map(lambda x: 1 if x == 1 else 0, valpattern))
            return
        for i, val in enumerate(valpattern):
            # generate a value tuple with a true value for each atom which is not set to false by evidence
            if val == 0:
                continue
            elif val is None:
                values = [0] * len(valpattern)
                values[i] = 1
                yield tuple(values)
        yield tuple([0] * len(atomindices))

    def value_index(self, value):
        if sum(value) > 1:
            raise Exception('Invalid world value for soft mutex block %s: %s' % (str(self), str(value)))
        elif sum(value) == 1:
            return value.index(1) + 1
        else:
            return 0
