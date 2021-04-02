import random
from collections import defaultdict

from dnutils import ProgressBar

from .mcmc import MCMCInference
from ..constants import HARD, ALL
from ..grounding.fastconj import FastConjunctionGrounding
from logic.elements import Logic


class SAMaxWalkSAT(MCMCInference):
    """
    A MaxWalkSAT MPE solver using simulated annealing.
    """
    
    def __init__(self, mrf, queries=ALL, state=None, **params):
        MCMCInference.__init__(self, mrf, queries, **params)
        if state is None:
            self.state = self.random_world(self.mrf.evidence)
        else:
            self.state = state
        self.sum = 0
        self.var2gf = defaultdict(set)
        self.weights = list(self.mrf.mln.weights)
        formulas = []
        for f in self.mrf.formulas:
            if f.weight < 0:
                f_ = self.mrf.mln.logic.negate(f)
                f_.weight = - f.weight
                formulas.append(f_.nnf())
        grounder = FastConjunctionGrounding(mrf, formulas=formulas, simplify=True, unsatfailure=True)
        for gf in grounder.iter_groundings():
            if isinstance(gf, Logic.TrueFalse): 
                continue
            vars_ = set([self.mrf.variable(a).index for a in gf.ground_atoms()])
            for v in vars_: 
                self.var2gf[v].add(gf)
            self.sum += (self.hardw if gf.weight == HARD else gf.weight) * (1 - gf(self.state))

    @property
    def thr(self):
        """
        threshold
        """
        return self._params.get('thr', 0)

    @property
    def hardw(self):
        """
        weight
        """
        return self._params.get('hardw', 10)

    @property
    def maxsteps(self):
        """
        max iteration
        """
        return self._params.get('maxsteps', 500)
     
    def _run(self):
        i = 0 
        i_max = self.maxsteps
        thr = self.thr
        if self.verbose:
            bar = ProgressBar(steps=i_max, color='green')
        while i < i_max and self.sum > self.thr:
            # randomly choose a variable to modify
            var = self.mrf.variables[random.randint(0, len(self.mrf.variables)-1)]
            evdict = var.value2dict(var.evidence_value(self.mrf.evidence))
            value_count = var.value_count(evdict) 
            if value_count == 1:   # this is evidence
                continue
            # compute the sum of relevant gf weights before the modification
            sum_before = 0
            for gf in self.var2gf[var.index]:
                sum_before += (self.hardw if gf.weight == HARD else gf.weight) * (1 - gf(self.state)) 
            # modify the state
            valindex = random.randint(0, value_count - 1)
            value = [v for _, v in var.iter_values(evdict)][valindex]
            oldstate = list(self.state)
            var.setval(value, self.state)
            # compute the sum after the modification
            sum_after = 0
            for gf in self.var2gf[var.index]:
                sum_after += (self.hardw if gf.weight == HARD else gf.weight) * (1 - gf(self.state))
            # determine whether to keep the new state            
            keep = False
            improvement = sum_after - sum_before
            if improvement < 0 or sum_after <= thr: 
                prob = 1.0
                keep = True
            else: 
                prob = (1.0 - min(1.0, abs(improvement / self.sum))) * (1 - (float(i) / i_max))
                keep = random.uniform(0.0, 1.0) <= prob
#                 keep = False # !!! no annealing
            # apply new objective value
            if keep:
                self.sum += improvement
            else:
                self.state = oldstate
            # next iteration
            i += 1
            if self.verbose:
                bar.label('sum = %f' % self.sum)
                bar.inc()
        if self.verbose:
            print("SAMaxWalkSAT: %d iterations, sum=%f, threshold=%f" % (i, self.sum, self.thr))
        self.mrf.mln.weights = self.weights
        return dict([(str(q), self.state[q.ground_atom.index]) for q in self.queries])
