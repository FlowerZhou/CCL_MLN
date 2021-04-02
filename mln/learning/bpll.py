from dnutils import logs
from dnutils.console import barstr

from .common import AbstractLearner
from collections import defaultdict
import numpy
from mln.util import fsum, temporary_evidence
from numpy.ma.core import sqrt, log
from mln.grounding.default import DefaultGroundingFactory
from mln.grounding.fastconj import FastConjunctionGrounding
from mln.mrfvars import SoftMutexVariable
from mln.learning.common import DiscriminativeLearner
from mln.grounding.bpll import BPLLGroundingFactory
from mln.constants import HARD
import pdb
from numba import jit

logger = logs.getlogger(__name__)

grounder = None


class BPLL(AbstractLearner):
    """
    Pseudo-log-likelihood learning with blocking, i.e. a generalization
    of PLL which takes into consideration the fact that the truth value of a
    blocked atom cannot be inverted without changing a further atom's truth
    value from the same block.
    This learner is fairly efficient, as it computes f and grad based only
    on a sufficient statistic.
    """

    def __init__(self, mrf, **params):
        AbstractLearner.__init__(self, mrf, **params)
        self._pls = None
        self._stat = None  # {formula_index: {var_index: [val_index_truth_num, val_index_truth_num]}}
        self._var_index2f_index = None  # record the [var_index]->[formula_index], ground_formula is true
        self._lastw = None

    def _prepare(self):
        logger.debug("computing statistics...")
        self._compute_statistics()
        # pdb.set_trace()

        # print self._stat

    def _pl(self, var_index, w):
        """
        Computes the pseudo-likelihoods for the given variable under weights w.
        """
        var = self.mrf.variable(var_index)
        values = var.value_count()
        gfs = self._var_index2f_index.get(var_index)
        if gfs is None:  # no list was saved, so the truth of all formulas is unaffected by the variable's value
            # uniform distribution applies
            p = 1.0 / values
            return [p] * values
        sums = [0] * values  # numpy.zeros(values)
        for f_index in gfs:
            for val_index, n in enumerate(self._stat[f_index][var_index]):
                if w[f_index] == HARD:
                    # set the prob mass of every value violating a hard constraint to None
                    # to indicate a globally inadmissible value. We will set those ones to 0 afterwards.
                    if n == 0:
                        sums[val_index] = None
                elif sums[val_index] is not None:
                    # don't set it if this value has already been assigned marked as inadmissible.
                    sums[val_index] += n * w[f_index]
        exp_sums = [numpy.exp(s) if s is not None else 0 for s in sums]  # numpy.exp(numpy.array(sums))
        z = sum(exp_sums)
        return [w_ / z for w_ in exp_sums]

    def write_pls(self):
        for var in self.mrf.variables:
            print(repr(var))
            for i, value in var.iter_values():
                print('    ', barstr(color='magenta', percent=self._pls[var.index][i]) + (
                    '*' if var.evidence_value_index() == i else ' '), i, value)

    def _compute_pls(self, w):
        if self._pls is None or self._lastw is None or self._lastw != list(w):
            self._pls = [self._pl(var.index, w) for var in self.mrf.variables]
            self._lastw = list(w)

    def _f(self, w):
        self._compute_pls(w)
        probs = []
        for var in self.mrf.variables:
            p = self._pls[var.index][var.evidence_value_index()]
            if p == 0:
                p = 1e-10  # prevent 0 probabilities
            probs.append(p)
        return fsum(list(map(log, probs)))

    def _grad(self, w):
        self._compute_pls(w)
        grad = numpy.zeros(len(self.mrf.formulas), numpy.float64)
        for f_index, var_val in self._stat.items():
            for var_index, counts in var_val.items():
                ev_index = self.mrf.variable(var_index).evidence_value_index()
                g = counts[ev_index]
                for i, val in enumerate(counts):
                    g -= val * self._pls[var_index][i]
                grad[f_index] += g
        self.grad_opt_norm = sqrt(float(fsum([x * x for x in grad])))
        return numpy.array(grad)

    def _add_stat(self, f_index, var_index, val_index, inc=1):
        if f_index not in self._stat:
            self._stat[f_index] = {}
        d = self._stat[f_index]
        if var_index not in d:
            d[var_index] = [0] * self.mrf.variable(var_index).value_count()
        d[var_index][val_index] += inc

    def _compute_statistics(self):
        """
        computes the statistics upon which the optimization is based
        """
        self._stat = {}
        self._var_index2f_index = defaultdict(set)
        # pdb.set_trace()
        # print("Possible world is %d " % self.mrf.count_worlds())
        global  grounder
        grounder = DefaultGroundingFactory(self.mrf, simplify=False, unsatfailure=True, verbose=self.verbose, cache=0)

        print("multicore is %s" % grounder.multicore)
        # pdb.set_trace()
        for f in grounder.iter_groundings():
            for ground_atom in f.ground_atoms():
                var = self.mrf.variable(ground_atom)
                with temporary_evidence(self.mrf):
                    for val_index, value in var.iter_values():
                        var.setval(value, self.mrf.evidence)
                        truth = f(self.mrf.evidence)
                        if truth != 0:
                            self._var_index2f_index[var.index].add(f.index)
                            self._add_stat(f.index, var.index, val_index, truth)


class DPLL(BPLL, DiscriminativeLearner):
    
    # Discriminative pseudo-log-likelihood learning.

    def _f(self, w, **params):
        self._compute_pls(w)
        probs = []
        for var in self.mrf.variables:
            if var.predicate.name in self.epreds:
                continue
            p = self._pls[var.index][var.evidence_value_index()]
            if p == 0: 
                p = 1e-10  # prevent 0 probabilities
            probs.append(p)
        return fsum(list(map(log, probs)))

    def _grad(self, w, **params):
        self._compute_pls(w)
        grad = numpy.zeros(len(self.mrf.formulas), numpy.float64)
        for f_index, var_val in self._stat.items():
            for var_index, counts in var_val.items():
                if self.mrf.variable(var_index).predicate.name in self.epreds:
                    continue
                ev_index = self.mrf.variable(var_index).evidence_value_index()
                g = counts[ev_index]
                for i, val in enumerate(counts):
                    g -= val * self._pls[var_index][i]
                grad[f_index] += g
        self.grad_opt_norm = sqrt(float(fsum([x * x for x in grad])))
        return numpy.array(grad)


class BPLL_CG(BPLL):

    def _prepare(self):
        global grounder
        grounder = BPLLGroundingFactory(self.mrf, multicore=self.multicore, verbose=self.verbose)
        for _ in grounder.iter_groundings():
            pass
        self._stat = grounder._stat
        self._var_index2f_index = grounder._var_index2f_index


class DBPLL_CG(DPLL):

    def _prepare(self):
        global grounder
        grounder = BPLLGroundingFactory(self.mrf, multicore=self.multicore, verbose=self.verbose)
        for _ in grounder.iter_groundings():
            pass
        self._stat = grounder._stat
        self._var_index2f_index = grounder._var_index2f_index
