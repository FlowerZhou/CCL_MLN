import random
from collections import defaultdict

import numpy
from dnutils import ProgressBar, logs

from logic.elements import Logic
from mln.constants import ALL
from mln.grounding.fastconj import FastConjunctionGrounding
from mln.inference.mcmc import MCMCInference
import pdb
from utils.multicore import check_mem
from multiprocessing import Pool
from utils.multicore import with_tracing

var_ = None
var2gf_ = None
evidence_ = None
logger = logs.getlogger(__name__)

"""
def _value_probs_multi(world):
    sums = [0] * var_.value_count()
    for gf in var2gf_[var_.index]:
        possible_values = []
        for i, value in var_.iter_values(evidence_):
            possible_values.append(i)
            world_ = var_.setval(value, list(world))
            truth = gf(world_)
            if truth == 0 and gf.is_hard:
                sums[i] = None
            elif sums[i] is not None and not gf.is_hard:
                sums[i] += gf.weight * truth
        # set all impossible values to None (i.e. prob 0) since they
        # might still be have a value of 0 in sums
        for i in [j for j in range(len(sums)) if j not in possible_values]:
            sums[i] = None

    exp_sums = numpy.array([numpy.exp(s) if s is not None else 0 for s in sums])
    z = sum(exp_sums)
    probs = exp_sums / z
    return probs
"""


def cal_chain(chain):
    check_mem()
    print("enter cal_chain")
    chain.step()


class GibbsSampler(MCMCInference):

    def __init__(self, mrf, queries=ALL, **params):
        MCMCInference.__init__(self, mrf, queries, **params)
        self.var2gf = defaultdict(set)
        # pdb.set_trace()
        grounder = FastConjunctionGrounding(mrf, simplify=True, unsatfailure=True, cache=None)
        for gf in grounder.iter_groundings():
            if isinstance(gf, Logic.TrueFalse):
                continue
            vars_ = set([self.mrf.variable(a).index for a in gf.ground_atoms()])
            for v in vars_:
                self.var2gf[v].add(gf)

    @property
    def chains(self):
        return self._params.get('chains', 1)

    @property
    def maxsteps(self):
        return self._params.get('maxsteps', 100)

    class Chain(MCMCInference.Chain):

        def __init__(self, infer, queries):
            MCMCInference.Chain.__init__(self, infer, queries)
            mrf = infer.mrf

        def _value_probs(self, var, world):
            # pdb.set_trace()
            sums = [0] * var.value_count()
            for gf in self.infer.var2gf[var.index]:
                possible_values = []
                for i, value in var.iter_values(self.infer.mrf.evidence):
                    possible_values.append(i)
                    world_ = var.setval(value, list(world))
                    truth = gf(world_)
                    if truth == 0 and gf.is_hard:
                        sums[i] = None
                    elif sums[i] is not None and not gf.is_hard:
                        sums[i] += gf.weight * truth
                # set all impossible values to None (i.e. prob 0) since they
                # might still be have a value of 0 in sums
                for i in [j for j in range(len(sums)) if j not in possible_values]:
                    sums[i] = None

            exp_sums = numpy.array([numpy.exp(s) if s is not None else 0 for s in sums])
            z = sum(exp_sums)
            probs = exp_sums / z
            return probs

        def step(self):
            # pdb.set_trace()
            mrf = self.infer.mrf
            # reassign values by sampling from the conditional distributions given the Markov blanket
            state = list(self.state)
            pdb.set_trace()
            for var in mrf.variables:
                # compute distribution to sample from
                values = list(var.values())
                if len(values) == 1:  # do not sample if we have evidence
                    continue
                probs = self._value_probs(var, self.state)
                # self._value_probs(var, self.state)
                # pool = Pool()
                # for probs in pool.map(_value_probs_multi, self.state):
                # check for soft evidence and greedily satisfy it if possible
                index = None
                # sample value
                if index is None:
                    r = random.uniform(0, 1)
                    index = 0
                    s = probs[0]
                    while r > s:
                        index += 1
                        s += probs[index]
                var.setval(values[index], self.state)
            # update results
            self.update(self.state)

    def _run(self, **params):
        """
        infer one or more probabilities P(F1 | F2)
        what: a ground formula (string) or a list of ground formulas (list of strings) (F1)
        given: a formula as a string (F2)
        set evidence according to given conjunction (if any)
        """
        # initialize chains
        chains = MCMCInference.ChainGroup(self)
        for i in range(self.chains):
            chain = GibbsSampler.Chain(self, self.queries)
            chains.chain(chain)
        # do Gibbs sampling
        # if verbose and details: print "sampling..."
        # pdb.set_trace()
        converged = 0
        steps = 0
        if self.verbose:
            bar = ProgressBar(color='green', steps=self.maxsteps)
        while converged != self.chains and steps < self.maxsteps:
            converged = 0
            steps += 1
            # pool = Pool(maxtasksperchild=1)
            for chain in chains.chains:
                chain.step()
            if self.verbose:
                bar.inc()
                bar.label('%d / %d' % (steps, self.maxsteps))
            """
            try:
                pool.imap(with_tracing(cal_chain), chains.chains)

                # for chain in chains.chains:
                # chain.step()
                if self.verbose:
                    bar.inc()
                    bar.label('%d / %d' % (steps, self.maxsteps))
            except Exception as e:
                logger.error('Error in child process. Terminating pool...')
                pool.close()
                raise e
            finally:
                pool.terminate()
                pool.join()
            """

        return chains.results()[0]
