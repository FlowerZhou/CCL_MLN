import random

from dnutils import logs

from mln.inference.infer import Inference
from mln.util import fstr
from mln.constants import ALL
import pdb
from numba import jit

logger = logs.getlogger(__name__)

active_variables = []    # record active variables' index


class MCMCInference(Inference):
    """
    Abstract super class for Markov chain Monte Carlo-based inference.
    """

    def __init__(self, mrf, queries=ALL, **params):
        Inference.__init__(self, mrf, queries, **params)

    def random_world(self, evidence=None):
        """
        Get a random possible world, taking the evidence into account.
        """
        if evidence is None:
            world = list(self.mrf.evidence)
        else:
            world = list(evidence)
        for var in self.mrf.variables:
            # pdb.set_trace()
            evdict = var.value2dict(var.evidence_value(world))
            value_count = var.value_count(evdict)
            if value_count > 1:
                # get a random value of the variable
                validx = random.randint(0, value_count - 1)
                value = [v for _, v in var.iter_values(evdict)][validx]
                var.setval(value, world)
        # pdb.set_trace()
        return world

    def smart_random_world(self, evidence=None):
        """
        lazy initialization for SampleSAT
        """
        # print("Enter smart random world! ")
        k = 1   # find a k-neighbor of one variable to another
        if evidence is None:
            world = list(self.mrf.evidence)
        else:
            world = list(evidence)
        for var in self.mrf.variables:
            evdict = var.value2dict(var.evidence_value(world))
            value_count = var.value_count(evdict)
            if value_count > 1:
                validx = random.randint(0, value_count - 1)
                value = [v for _, v in var.iter_values(evdict)][validx]
                if value[0] is 1 and var.index not in active_variables:
                    active_variables.append(var.index)
                var.setval(value, world)
        # pdb.set_trace()
        return world

    class Chain:
        """
        Represents the state of a Markov Chain.
        """

        def __init__(self, infer, queries):
            self.queries = queries
            self.soft_evidence = None
            self.steps = 0
            self.truths = [0] * len(self.queries)
            self.converged = False
            self.last_result = 10
            self.infer = infer
            # copy the current  evidence as this chain's state
            # initialize remaining variables randomly (but consistently with the evidence)
            self.state = infer.random_world()
            # self.state = infer.smart_random_world()
        """
        def __getstate__(self):
            d = self.__dict__
            return d

        def __setstate__(self, d):
            self.__dict__ = d
        """

        def update(self, state):
            # pdb.set_trace()
            self.steps += 1
            self.state = state
            # keep track of counts for queries
            for i, q in enumerate(self.queries):
                self.truths[i] += q(self.state)
            # check if converged !!! TODO check for all queries
            if self.steps % 50 == 0:
                result = self.results()[0]
                diff = abs(result - self.last_result)
                if diff < 0.001:
                    self.converged = True
                self.last_result = result
            # keep track of counts for soft evidence
            if self.soft_evidence is not None:
                for se in self.soft_evidence:
                    self.softev_counts[se["expr"]] += se["formula"](self.state)

        def set_soft_evidence(self, soft_evidence):
            self.soft_evidence = soft_evidence
            self.softev_counts = {}
            for se in soft_evidence:
                if 'formula' not in se:
                    formula = self.infer.mrf.mln.logic.parse_formula(se['expr'])
                    se['formula'] = formula.ground(self.infer.mrf, {})
                    se['expr'] = fstr(se['formula'])
                self.softev_counts[se["expr"]] = se["formula"](self.state)

        def soft_evidence_frequency(self, formula):
            if self.steps == 0:
                return 0
            return float(self.softev_counts[fstr(formula)]) / self.steps

        def results(self):
            results = []
            # pdb.set_trace()
            for i in range(len(self.queries)):
                results.append(float(self.truths[i]) / self.steps)
            return results

    class ChainGroup:

        def __init__(self, infer):
            self.chains = []
            self.infer = infer

        def chain(self, chain):
            self.chains.append(chain)

        def results(self):
            chains = float(len(self.chains))
            queries = self.chains[0].queries
            # compute average
            results = [0.0] * len(queries)
            for chain in self.chains:
                cr = chain.results()
                for i in range(len(queries)):
                    results[i] += cr[i] / chains
            # compute variance
            var = [0.0 for i in range(len(queries))]
            for chain in self.chains:
                cr = chain.results()
                for i in range(len(self.chains[0].queries)):
                    var[i] += (cr[i] - results[i]) ** 2 / chains
            return dict([(str(q), p) for q, p in zip(queries, results)]), var

        def avgtruth(self, formula):
            # returns the fraction of chains in which the given formula is currently true '''
            t = 0.0
            for c in self.chains:
                t += formula(c.state)
            return t / len(self.chains)
