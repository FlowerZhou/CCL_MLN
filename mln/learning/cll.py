from dnutils import logs

from mln.learning.common import AbstractLearner, DiscriminativeLearner
import random
from collections import defaultdict
from mln.util import fsum, dict_union, temporary_evidence
from numpy.ma.core import log, sqrt
import numpy
from logic.elements import Logic
from mln.constants import HARD


logger = logs.getlogger(__name__)


class CLL(AbstractLearner):
    """
    Implementation of composite-log-likelihood learning.
    """

    def __init__(self, mrf, **params):
        AbstractLearner.__init__(self, mrf, **params)
        self.partitions = []
        self.repart = 0

    @property
    def partsize(self):
        return self._params.get('partsize', 1)

    @property
    def maxiter(self):
        return self._params.get('maxiter', 10)

    @property
    def variables(self):
        return self.mrf.variables

    def _prepare(self):
        # create random partition of the ground atoms
        self.partitions = []
        self.atomindex2partition = {}
        self.partition2formulas = defaultdict(set)
        self.evindex = {}
        self.value_counts = {}
        self.partitionProbValues = {}
        self.current_wts = None
        self.iter = 0
        self.probs = {}
        self._stat = {}
        size = self.partsize
        variables = list(self.variables)
        if size > 1:
            random.shuffle(variables)
        while len(variables) > 0:
            vars_ = variables[:size if len(variables) > size else len(variables)]
            partindex = len(self.partitions)
            partition = CLL.Partition(self.mrf, vars_, partindex)
            # create the mapping from atoms to their partitions
            for atom in partition.ground_atoms:
                self.atomindex2partition[atom.index] = partition
            logger.debug('created partition: %s' % str(partition))
            self.value_counts[partindex] = partition.value_count()
            self.partitions.append(partition)
            self.evindex[partindex] = partition.evidenceindex()
            variables = variables[len(partition.variables):]
        logger.debug('CLL created %d partitions' % len(self.partitions))
        self._compute_statistics()

    def repeat(self):
        return True

    def _addstat(self, findex, pindex, valindex, inc=1):
        if findex not in self._stat:
            self._stat[findex] = {}
        d = self._stat[findex]
        if pindex not in d:
            d[pindex] = [0] * self.value_counts[pindex]
        try:
            d[pindex][valindex] += inc
        except Exception as e:
            raise e

    def _compute_statistics(self):
        self._stat = {}
        self.partition2formulas = defaultdict(set)
        for formula in self.mrf.formulas:
            literals = []
            for lit in formula.literals():
                literals.append(lit)
            # in case of a conjunction, rearrange the literals such that
            # equality constraints are evaluated first
            isconj = self.mrf.mln.logic.is_literal_conj(formula)
            if isconj:
                literals = sorted(literals, key=lambda l: -1 if isinstance(l, Logic.Equality) else 1)
            self._compute_stat_rec(literals, [], {}, formula, isconj=isconj)

    def _compute_stat_rec(self, literals, gndliterals, var_assign, formula, f_gndlit_parts=None, processed=None,
                          isconj=False):
        """
        TODO: make sure that there are no equality constraints in the conjunction!
        """

        if len(literals) == 0:
            # at this point, we have a fully grounded conjunction in ground_literals
            # create a mapping from a partition to the ground literals in this formula
            # (criterion no. 1, applies to all kinds of formulas)
            part2gndlits = defaultdict(list)
            part_with_f_lit = None
            for gndlit in gndliterals:
                if isinstance(gndlit, Logic.Equality) or hasattr(self,
                                                                 'qpreds') and gndlit.ground_atom.predname not in self.qpreds: continue
                part = self.atomindex2partition[gndlit.ground_atom.index]
                part2gndlits[part].append(gndlit)
                if gndlit(self.mrf.evidence) == 0:
                    part_with_f_lit = part

            # if there is a false ground literal we only need to take into account
            # the partition comprising this literal (criterion no. 2)
            # there is maximally one such partition with false literals in the conjunction
            # because of criterion no. 5
            if isconj and part_with_f_lit is not None:
                gndlits = part2gndlits[part_with_f_lit]
                part2gndlits = {part_with_f_lit: gndlits}
            if not isconj:  # if we don't have a conjunction, ground the formula with the given variable assignment
                # print 'formula', formula
                gndformula = formula.ground(self.mrf, var_assign)
                # print 'gndformula', gndformula
                # stop()
            for partition, gndlits in part2gndlits.items():
                # for each partition, select the ground atom truth assignments
                # in such a way that the conjunction is rendered true. There
                # is precisely one such assignment for each partition. (criterion 3/4)
                evidence = {}
                if isconj:
                    for gndlit in gndlits:
                        evidence[gndlit.ground_atom.index] = 0 if gndlit.negated else 1
                for world in partition.iter_values(evidence):
                    # update the sufficient statistics for the given formula, partition and world value
                    worldindex = partition.value_index(world)
                    if isconj:
                        truth = 1
                    else:
                        # temporarily set the evidence in the MRF, compute the truth value of the
                        # formula and remove the temp evidence
                        with temporary_evidence(self.mrf):
                            for atomindex, value in partition.value2dict(world).items():
                                self.mrf.set_evidence({atomindex: value}, erase=True)
                            truth = gndformula(self.mrf.evidence)
                            if truth is None:
                                print(gndformula)
                                print(gndformula.print_structure(self.mrf.evidence))

                    if truth != 0:
                        self.partition2formulas[partition.index].add(formula.index)
                        self._addstat(formula.index, partition.index, worldindex, truth)
            return

        lit = literals[0]
        # ground the literal with the existing assignments
        gndlit = lit.ground(self.mrf, var_assign, partial=True)
        for assign in Logic.iter_eq_varassignments(gndlit, formula, self.mrf) if isinstance(gndlit,
                                                                                            Logic.Equality) else gndlit.iter_var_groundings(
                self.mrf):
            # copy the arguments to avoid side effects
            # if f_gndlit_parts is None: f_gndlit_parts = set()
            # else: f_gndlit_parts = set(f_gndlit_parts)
            if processed is None:
                processed = []
            else:
                processed = list(processed)
            # ground with the remaining free variables
            gndlit_ = gndlit.ground(self.mrf, assign)
            truth = gndlit_(self.mrf.evidence)
            # treatment of equality constraints
            if isinstance(gndlit_, Logic.Equality):
                if isconj:
                    if truth == 1:
                        self._compute_stat_rec(literals[1:], gndliterals, dict_union(var_assign, assign), formula,
                                               f_gndlit_parts, processed, isconj)
                    else:
                        continue
                else:
                    self._compute_stat_rec(literals[1:], gndliterals + [gndlit_], dict_union(var_assign, assign),
                                           formula, f_gndlit_parts, processed, isconj)
                continue
            atom = gndlit_.ground_atom

            if atom.index in processed:
                continue

            # if we encounter a gnd literal that is false by the evidence
            # and there is already a false one in this grounding from a different
            # partition, we can stop the grounding process here. The gnd conjunction
            # will never ever be rendered true by any of this partitions values (criterion no. 5)
            isevidence = hasattr(self, 'qpreds') and gndlit_.ground_atom.predname not in self.qpreds
            # assert isEvidence == False
            if isconj and truth == 0:
                if f_gndlit_parts is not None and atom not in f_gndlit_parts:
                    continue
                elif isevidence:
                    continue
                else:
                    self._compute_stat_rec(literals[1:], gndliterals + [gndlit_], dict_union(var_assign, assign),
                                           formula, self.atomindex2partition[atom.index], processed, isconj)
                    continue
            elif isconj and isevidence:
                self._compute_stat_rec(literals[1:], gndliterals, dict_union(var_assign, assign), formula,
                                       f_gndlit_parts, processed, isconj)
                continue

            self._compute_stat_rec(literals[1:], gndliterals + [gndlit_], dict_union(var_assign, assign), formula,
                                   f_gndlit_parts, processed, isconj)

    def _compute_probs(self, w):
        probs = {}  # numpy.zeros(len(self.partitions))
        for pindex in range(len(self.partitions)):
            expsums = [0] * self.value_counts[pindex]
            for findex in self.partition2formulas[pindex]:
                for i, v in enumerate(self._stat[findex][pindex]):
                    if w[findex] == HARD:
                        if v == 0: expsums[i] = None
                    elif expsums[i] is not None:
                        #                         out('adding', v, '*', w[findex], 'to', i)
                        expsums[i] += v * w[findex]
            expsum = numpy.array(
                [numpy.exp(s) if s is not None else 0 for s in expsums])  # leave out the inadmissible values
            z = fsum(expsum)
            if z == 0:
                raise Exception('MLN is unsatisfiable: all probability masses of partition %s are zero.' % str(self.partitions[pindex]))
            probs[pindex] = expsum / z
            self.probs[pindex] = expsum
        self.probs = probs
        return probs

    def _f(self, w):
        if self.current_wts is None or list(w) != self.current_wts:
            self.current_wts = list(w)
            self.probs = self._compute_probs(w)
        likelihood = numpy.zeros(len(self.partitions))
        for pindex in range(len(self.partitions)):
            p = self.probs[pindex][self.evindex[pindex]]
            if p == 0: p = 1e-10
            likelihood[pindex] += p
        self.iter += 1
        return fsum(list(map(log, likelihood)))

    def _grad(self, w, **params):
        if self.current_wts is None or not list(w) != self.current_wts:
            self.current_wts = w
            self.probs = self._compute_probs(w)
        grad = numpy.zeros(len(w))
        for findex, partitions in self._stat.items():
            for part, values in partitions.items():
                v = values[self.evindex[part]]
                for i, val in enumerate(values):
                    v -= self.probs[part][i] * val
                grad[findex] += v
        self.grad_opt_norm = sqrt(float(fsum([x * x for x in grad])))
        return numpy.array(grad)

    class Partition(object):
        """
        Represents a partition of the variables in the MRF. Provides a couple
        of convenience methods.
        """

        def __init__(self, mrf, variables, index):
            self.variables = variables
            self.mrf = mrf
            self.index = index

        @property
        def ground_atoms(self):
            atoms = []
            for v in self.variables:
                atoms.extend(v.ground_atoms)
            return atoms

        def __contains__(self, atom):
            """
            Returns True if the given ground atom or ground atom index is part of
            this partition.
            """
            if isinstance(atom, Logic.GroundAtom):
                return atom in self.ground_atoms
            elif type(atom) is int:
                return self.mrf.ground_atom(atom) in self
            else:
                raise Exception('Invalid type of atom: %s' % type(atom))

        def value2dict(self, value):
            """
            Takes a possible world tuple of the form ((0,),(0,),(1,0,0),(1,)) and transforms
            it into a dict mapping the respective atom indices to their truth values
            """
            evidence = {}
            for var, val in zip(self.variables, value):
                evidence.update(var.value2dict(val))
            return evidence

        def evidenceindex(self, evidence=None):
            """
            Returns the index of the possible world value of this partition that is represented
            by evidence. If evidence is None, the evidence set in the MRF is used.
            """
            if evidence is None:
                evidence = self.mrf.evidence
            evidencevalue = []
            for var in self.variables:
                evidencevalue.append(var.evidence_value(evidence))
            return self.value_index(tuple(evidencevalue))

        def value_index(self, value):
            """
            Computes the index of the given possible world that would be assigned
            to it by recursively generating all worlds by iter_values().
            value needs to be represented by a (nested) tuple of truth values.
            Exp: ((0,),(0,),(1,0,0),(0,)) --> 0
                 ((0,),(0,),(1,0,0),(1,)) --> 1
                 ((0,),(0,),(0,1,0),(0,)) --> 2
                 ((0,),(0,),(0,1,0),(1,)) --> 3
                 ...
            """
            index = 0
            for i, (var, val) in enumerate(zip(self.variables, value)):
                exponential = 2 ** (len(self.variables) - i - 1)
                valindex = var.value_index(val)
                index += valindex * exponential
            return index

        def iter_values(self, evidence=None):
            """
            Yields possible world values of this partition in the form
            ((0,),(0,),(1,0,0),(0,)), for instance. Nested tuples represent mutex variables.
            All tuples are consistent with the evidence at hand. Evidence is
            a dict mapping a ground atom index to its (binary) truth value.
            """
            if evidence is None:
                evidence = []
            for world in self._iter_values(self.variables, [], evidence):
                yield world

        def _iter_values(self, variables, assignment, evidence):
            """
            Recursively generates all tuples of possible worlds that are consistent
            with the evidence at hand.
            """
            if not variables:
                yield tuple(assignment)
                return
            var = variables[0]
            for _, val in var.iter_values(evidence):
                for world in self._iter_values(variables[1:], assignment + [val], evidence):
                    yield world

        def value_count(self):
            """
            Returns the number of possible (partial) worlds of this partition
            """
            count = 1
            for v in self.variables:
                count *= v.value_count()
            return count

        def __str__(self):
            s = []
            for v in self.variables:
                s.append(str(v))
            return '%d: [%s]' % (self.index, ','.join(s))


class DCLL(CLL, DiscriminativeLearner):
    """
    Discriminative Composite-Likelihood Learner.
    """

    def __init__(self, mrf=None, **params):
        CLL.__init__(self, mrf, **params)

    @property
    def variables(self):
        return [var for var in self.mrf.variables if var.predicate.name in self.qpreds]


Partition = CLL.Partition
