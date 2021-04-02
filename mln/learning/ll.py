import sys

from dnutils import ProgressBar
from .common import *
from mln.mrfvars import SoftMutexVariable
from mln.grounding.default import DefaultGroundingFactory
from mln.constants import HARD


class LL(AbstractLearner):
    """
    Exact Log-Likelihood learner.
    """
    def __init__(self, mrf, **params):
        AbstractLearner.__init__(self, mrf, **params)
        self._stat = None
        self._ls = None
        self._eworld_index = None
        self._lastw = None

    def _prepare(self):
        self._compute_statistics()

    def _l(self, w):
        """
        computes the likelihoods of all possible worlds under weights w
        """
        if self._lastw is None or list(w) != self._lastw:
            self._lastw = list(w)
            expsums = []
            for fvalues in self._stat:
                s = 0
                hc_violation = False
                for findex, val in fvalues.items():
                    if self.mrf.mln.formulas[findex].weight == HARD:
                        if val == 0:
                            hc_violation = True
                            break
                    else:
                        s += val * w[findex]
                if hc_violation:
                    expsums.append(0)
                else:
                    expsums.append(exp(s))
            z = sum(expsums)
            if z == 0:
                raise Exception('MLN is unsatisfiable: probability masses of all possible worlds are zero.')
            self._ls = [v / z for v in expsums]
        return self._ls

    def _f(self, w):
        ls = self._l(w)
        return numpy.log(ls[self._eworld_index])

    def _grad(self, w):
        ls = self._l(w)
        grad = numpy.zeros(len(self.mrf.formulas), numpy.float64)
        for windex, values in enumerate(self._stat):
            for findex, count in values.items():
                if windex == self._eworld_index:
                    grad[findex] += count
                grad[findex] -= count * ls[windex]
        return grad

    def _compute_statistics(self):
        self._stat = []
        grounder = DefaultGroundingFactory(self.mrf)
        eworld = list(self.mrf.evidence)
        if self.verbose:
            bar = ProgressBar(width=100, steps=self.mrf.countworlds(), color='green')
        for windex, world in self.mrf.iterallworlds():
            if self.verbose:
                bar.label(str(windex))
                bar.inc()
            values = {}
            self._stat.append(values)
            if self._eworld_index is None and world == eworld:
                self._eworld_index = windex
            for gf in grounder.itergroundings():
                truth = gf(world)
                if truth != 0: values[gf.index] = values.get(gf.index, 0) + truth
