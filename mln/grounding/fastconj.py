from dnutils import logs, ProgressBar

from mln.grounding.default import DefaultGroundingFactory
from logic.elements import Logic
import types
from multiprocessing.pool import Pool
from utils.multicore import with_tracing
from mln.mlnpreds import FunctionalPredicate, SoftFunctionalPredicate, FuzzyPredicate
from mln.util import dict_union, rndbatches, cumsum
from mln.constants import HARD
from collections import defaultdict
from functools import reduce, partial
import pdb
import pudb
import pickle
import json

logger = logs.getlogger(__name__)

# this readonly global is for multiprocessing to exploit copy-on-write on linux systems
global_fastConjGrounding = None


# multiprocessing function
def create_formula_groundings(formulas):
    gfs = []
    # print("enter create formula groundings")
    # pdb.set_trace()

    for formula in sorted(formulas, key=global_fastConjGrounding._fsort):
        if global_fastConjGrounding.mrf.mln.logic.is_literal_conj(formula) or global_fastConjGrounding.mrf.mln.logic.\
                is_clause(formula):   # if formula is clause-form, can be pruned
            # print("enter fast grounding")
            for gf in global_fastConjGrounding.iter_groundings_fast(formula):
                gfs.append(gf)
        else:
            for gf in formula.iter_groundings(global_fastConjGrounding.mrf, simplify=True):
                gfs.append(gf)

    return gfs


class FastConjunctionGrounding(DefaultGroundingFactory):
    """
    Fairly fast grounding of conjunctions pruning the grounding tree if a
    formula is rendered false by the evidence. Performs some heuristic
    sorting such that equality constraints are evaluated first.
    """

    def __init__(self, mrf, simplify=False, unsatfailure=False, formulas=None,
                 cache=None, **params):
        DefaultGroundingFactory.__init__(self, mrf, simplify=simplify, unsatfailure=unsatfailure, formulas=formulas, cache=cache, **params)

    def _conjsort(self, e):
        if isinstance(e, Logic.Equality):
            return 0.5
        elif isinstance(e, Logic.TrueFalse):
            return 1
        elif isinstance(e, Logic.GroundLiteral):
            if self.mrf.evidence[e.ground_atom.index] is not None:
                return 2
            elif type(self.mrf.mln.predicate(e.ground_atom.pred_name)) in (FunctionalPredicate, SoftFunctionalPredicate):
                return 3
            else:
                return 4
        elif isinstance(e, Logic.Literal) and type(
                self.mrf.mln.predicate(e.pred_name)) in (FunctionalPredicate, SoftFunctionalPredicate, FuzzyPredicate):
            return 5
        elif isinstance(e, Logic.Literal):
            return 6
        else:
            return 7

    @staticmethod
    def _fsort(f):
        if f.weight == HARD:
            return 0
        else:
            return 1

    @staticmethod
    def min_undef(*args):
        """
        Custom minimum function return None if one of its arguments
        is None and min(*args) otherwise.
        """
        if len([x for x in args if x == 0]) > 0:
            return 0
        return reduce(lambda x, y: None if (x is None or y is None) else min(x, y), args)

    @staticmethod
    def max_undef(*args):
        """
        Custom maximum function return None if one of its arguments
        is None and max(*args) otherwise.
        """
        if len([x for x in args if x == 1]) > 0:
            return 1
        # pdb.set_trace()
        return reduce(lambda x, y: None if x is None or y is None else max(x, y), args)

    def iter_groundings_fast(self, formula):
        """
        Recursively generate the groundings of a conjunction that do not
        have a definite truth value yet given the evidence.
        """
        # make a copy of the formula to avoid side effects
        formula = formula.ground(self.mrf, {}, partial=True, simplify=True)
        children = [formula] if not hasattr(formula, 'children') else formula.children
        # make equality constraints access their variable domains
        # this is a _really_ dirty hack but it does the job ;-)
        variables = formula.var_doms()

        def eqvardoms(self, v=None, c=None):
            if v is None:
                v = defaultdict(set)
            for a in self.args:
                if self.mln.logic.isvar(a):
                    v[a] = variables[a]
            return v

        for child in children:
            if isinstance(child, Logic.Equality):
                # replace the var_doms method in this equality instance by our customized one
                setattr(child, 'var_doms', types.MethodType(eqvardoms, child))
        literals = sorted(children, key=self._conjsort)
        truthpivot, pivotfct = (1, self.min_undef) if isinstance(formula, Logic.Conjunction) \
            else ((0, self.max_undef) if isinstance(formula, Logic.Disjunction) else (None, None))
        for gf in self._iter_groundings_fast(formula, literals, 0, pivotfct, truthpivot, {}):
            yield gf

    def _iter_groundings_fast(self, formula, constituents, cidx, pivotfct, truthpivot, assignment, level=0):
        if truthpivot == 0 and (isinstance(formula, Logic.Conjunction) or self.mrf.mln.logic.is_literal(formula)):
            if formula.weight == HARD:
                raise Exception('MLN is unsatisfiable given evidence due to hard constraint violation: {}'.format(str(formula)))
            return
        if truthpivot == 1 and (isinstance(formula, Logic.Disjunction) or self.mrf.mln.logic.is_literal(formula)):
            return
        if cidx == len(constituents):
            # we have reached the end of the formula constituents
            gf = formula.ground(self.mrf, assignment, simplify=True)
            if isinstance(gf, Logic.TrueFalse):
                return
            yield gf
            return
        c = constituents[cidx]
        for varass in c.iter_var_groundings(self.mrf, partial=assignment):
            newass = dict_union(assignment, varass)
            ga = c.ground(self.mrf, newass)
            truth = ga.truth(self.mrf.evidence)
            if truth is None:
                truthpivot_ = truthpivot
            elif truthpivot is None:
                truthpivot_ = truth
            else:
                # pdb.set_trace()
                truthpivot_ = pivotfct(truthpivot, truth)
            for gf in self._iter_groundings_fast(formula, constituents, cidx + 1, pivotfct, truthpivot_, newass, level + 1):
                yield gf

    def _iter_groundings(self, simplify=True, unsatfailure=True):
        # pdb.set_trace()
        # generate all groundings
        # print("Before multiprocessing......")
        # pudb.set_trace()
        if not self.formulas:
            return
        global global_fastConjGrounding
        global_fastConjGrounding = self
        batches = list(rndbatches(self.formulas, 20))
        batch_sizes = [len(b) for b in batches]
        if self.verbose:
            bar = ProgressBar(steps=sum(batch_sizes), color='green')
            i = 0
        # print("type is %s" % type(self))
        print(self.multicore)
        # print(type(batches), type(batches[0]), type(batches[0][0]))
        # batches = json.dumps(batches, default=lambda obj: obj.__dict__, sort_keys=True,indent=4)
        # batches = pickle.dumps(batches)
        if self.multicore:
            print("Enter Multicore")
            pool = Pool()
            # gfs = pool.imap(with_tracing(create_formula_groundings), batches)
            # gfs = list(gfs)
            # print(gfs)
            try:
                # print(batches)
                for gfs in pool.imap(create_formula_groundings, batches):
                    if self.verbose:
                        bar.inc(batch_sizes[i])
                        bar.label(str(cumsum(batch_sizes, i + 1)))
                        i += 1
                    for gf in gfs:
                        yield gf

            except Exception as e:
                logger.error('Error in child process. Terminating pool...')
                pool.close()
                raise e
            finally:
                pool.terminate()
                pool.join()
                print("fast multicore finished...")
        else:
            # pdb.set_trace()
            print("fast not in multicore...")
            for gfs in map(create_formula_groundings, batches):
                if self.verbose:
                    bar.inc(batch_sizes[i])
                    bar.label(str(cumsum(batch_sizes, i + 1)))
                    i += 1
                for gf in gfs:
                    yield gf

