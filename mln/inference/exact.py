from dnutils import logs, ProgressBar
from logic.fol import FirstOrderLogic
from multiprocessing import Pool
from mln.constants import auto, HARD
from mln.grounding.fastconj import FastConjunctionGrounding
from mln.util import Interval
from numpy.ma.core import exp
from mln.inference.infer import Inference
from utils.multicore import with_tracing
from logic.elements import Logic
from mln.mrfvars import FuzzyVariable
import pdb
from numba import jit
logger = logs.getlogger(__name__)

# this readonly global is for multiprocessing to exploit copy-on-write
# on linux systems
global_enumAsk = None


def eval_queries(world):
    """
    Evaluates the queries given a possible world.
    """
    numerators = [0] * len(global_enumAsk.queries)
    denominator = 0
    expsum = 0
    # pdb.set_trace()
    for gf in global_enumAsk.grounder.iter_groundings():
        if global_enumAsk.soft_evidence_formula(gf):
            # pdb.set_trace()
            expsum += gf.noisyor(world) * gf.weight
        else:
            # pdb.set_trace()
            truth = gf(world)
            if gf.weight == HARD:
                if truth in Interval(']0,1['):
                    raise Exception('No real-valued degrees of truth are allowed in hard constraints.')
                if truth == 1:
                    continue
                else:
                    return numerators, 0
            expsum += gf(world) * gf.weight
    expsum = exp(expsum)
    # update numerators
    for i, query in enumerate(global_enumAsk.queries):
        # pdb.set_trace()
        if query(world):
            numerators[i] += expsum
    denominator += expsum
    return numerators, denominator


class EnumerationAsk(Inference):
    """
    Inference based on enumeration of (only) the worlds compatible with the
    evidence; supports soft evidence (assuming independence)
    """

    def __init__(self, mrf, queries, **params):
        # pdb.set_trace()
        Inference.__init__(self, mrf, queries, **params)
        self.grounder = FastConjunctionGrounding(mrf, simplify=False, unsatfailure=False, formulas=mrf.formulas, cache=auto, verbose=False, multicore=False)
        for variable in self.mrf.variables:
            variable.consistent(self.mrf.evidence, strict=isinstance(variable, FuzzyVariable))
        # pdb.set_trace()

    def _run(self):
        """
        verbose: whether to print results (or anything at all, in fact)
        details: (given that verbose is true) whether to output additional
                 status information
        debug:   (given that verbose is true) if true, outputs debug
                 information, in particular the distribution over possible
                 worlds
        debugLevel: level of detail for debug mode
        """
        # check consistency with hard constraints:

        self._watch.tag('check hard constraints', verbose=self.verbose)
        hcgrounder = FastConjunctionGrounding(self.mrf, simplify=False, unsatfailure=True,
                                              formulas=[f for f in self.mrf.formulas if f.weight == HARD],
                                              **(self._params + {'multicore': False, 'verbose': False}))
        # re_variables = self.mrf.reduce_variables(self.mrf.variables, self.queries)
        self._watch.finish('check hard constraints')
        # compute number of possible worlds
        worlds = 1
        # pdb.set_trace()
        for variable in self.mrf.variables:
            #  for variable in re_variables:
            values = variable.value_count(self.mrf.evidence)
            worlds *= values
        numerators = [0.0 for i in range(len(self.queries))]
        denominator = 0.
        # pdb.set_trace()
        # start summing
        logger.debug("Summing over %d possible worlds..." % worlds)
        if worlds > 500000 and self.verbose:
            print('!!! %d WORLDS WILL BE ENUMERATED !!!' % worlds)
        k = 0
        print('!!! %d WORLDS WILL BE ENUMERATED !!!' % worlds)
        self._watch.tag('enumerating worlds', verbose=self.verbose)
        global global_enumAsk
        global_enumAsk = self
        bar = None
        if self.verbose:
            bar = ProgressBar(steps=worlds, color='green')
        # pdb.set_trace()
        if self.multicore:
            print("Enter exact multicore....")
            pool = Pool()
            try:
                for num, denum in pool.imap(with_tracing(eval_queries), self.mrf.worlds()):
                    denominator += denum
                    k += 1
                    for i, v in enumerate(num):
                        numerators[i] += v
                    if self.verbose:
                        bar.inc()
            except Exception as e:
                logger.error('Error in child process. Terminating pool...')
                pool.close()
                raise e
            finally:
                pool.terminate()
                pool.join()
                print("exact multi finished...")
        else:  # do it single core
            count_wd = 0
            for world in self.mrf.worlds():
                # compute exp. sum of weights for this world
                # pdb.set_trace()
                count_wd += 1
                num, denom = eval_queries(world)
                denominator += denom
                for i, _ in enumerate(self.queries):
                    numerators[i] += num[i]
                k += 1
                if self.verbose:
                    bar.update(float(k) / worlds)
        logger.debug("%d worlds enumerated" % k)
        self._watch.finish('enumerating worlds')
        if 'grounding' in self.grounder.watch.tags:
            self._watch.tags['grounding'] = self.grounder.watch['grounding']
        # normalize answers
        # print("calculate %d worlds " % count_wd)
        dist = map(lambda x: float(x) / denominator, numerators)
        result = {}
        for q, p in zip(self.queries, dist):
            result[str(q)] = p
        return result

    def soft_evidence_formula(self, gf):
        truths = [a.truth(self.mrf.evidence) for a in gf.ground_atoms()]
        if None in truths:
            return False
        return isinstance(self.mrf.mln.logic, FirstOrderLogic) and any([t in Interval('(0,1)') for t in truths])