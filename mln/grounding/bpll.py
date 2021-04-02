from collections import defaultdict
from dnutils import logs, ProgressBar
from .fastconj import FastConjunctionGrounding
from mln.util import dict_union
from mln.constants import HARD
from logic.elements import Logic
from utils.multicore import with_tracing, check_mem
import multiprocessing
from multiprocessing import Value

import pdb


import types
from multiprocessing.pool import Pool

# this readonly global is for multiprocessing to exploit copy-on-write
# on linux systems
global_bpll_grounding = None

logger = logs.getlogger(__name__)

# multiprocessing function


def create_formula_groundings(formula, unsatfailure=True):
    check_mem()
    results = []

    if global_bpll_grounding.mrf.mln.logic.is_literal_conj(formula):
        for res in global_bpll_grounding.iter_groundings_fast(formula):
            check_mem()
            results.append(res)
    else:
        for gf in formula.iter_groundings(global_bpll_grounding.mrf, simplify=False):
            check_mem()
            stat = []
            for ground_atom in gf.ground_atoms():
                world = list(global_bpll_grounding.mrf.evidence)
                var = global_bpll_grounding.mrf.variable(ground_atom)
                for val_index, value in var.iter_values():
                    var.setval(value, world)
                    truth = gf(world)
                    if truth != 0:
                        stat.append((var.index, val_index, truth))
                    elif unsatfailure and gf.weight == HARD and gf(global_bpll_grounding.mrf.evidence) != 1:
                        print()
                        gf.print_structure(global_bpll_grounding.mrf.evidence)
            results.append((gf.index, stat))
    return results


class BPLLGroundingFactory(FastConjunctionGrounding):
    """
    Grounding factory for efficient grounding of conjunctions for
    pseudo-likelihood learning.
    """

    def __init__(self, mrf, formulas=None, cache=None, **params):
        FastConjunctionGrounding.__init__(self, mrf, simplify=False, unsatfailure=False, formulas=formulas, cache=cache, **params)
        self._stat = {}
        self._var_index2f_index = defaultdict(set)

    def iter_groundings_fast(self, formula):
        """
        Recursively generate the groundings of a conjunction. Prunes the
        generated grounding tree in case that a formula cannot be rendered
        true by subsequent literals.
        """
        # make a copy of the formula to avoid side effects
        formula = formula.ground(self.mrf, {}, partial=True)
        children = [formula] if not hasattr(formula, 'children') else formula.children
        # make equality constraints access their variable domains
        # this is a _really_ dirty hack but it does the job ;-)
        var_doms = formula.var_doms()

        def eqvar_doms(self, v=None, c=None):
            if v is None:
                v = defaultdict(set)
            for a in self.args:
                if self.mln.logic.isvar(a):
                    v[a] = var_doms[a]
            return v

        for child in children:
            if isinstance(child, Logic.Equality):
                setattr(child, 'var_doms', types.MethodType(eqvar_doms, child))
        lits = sorted(children, key=self._conjsort)
        for gf in self._iter_groundings_fast(formula, lits, 0, assignment={}, variables=[]):
            yield gf

    def _iter_groundings_fast(self, formula, constituents, c_index, assignment, variables, false_var=None, level=0):
        if c_index == len(constituents):
            # no remaining literals to ground. return the ground formula
            # and statistics
            stat = [(var_index, val_index, count) for (var_index, val_index, count) in variables]
            yield formula.index, stat
            return
        c = constituents[c_index]
        # go through all remaining groundings of the current constituent
        for varass in c.iter_var_groundings(self.mrf, partial=assignment):
            gnd = c.ground(self.mrf, dict_union(varass, assignment))
            # check if it violates a hard constraint
            if formula.weight == HARD and gnd(self.mrf.evidence) < 1:
                raise Exception('MLN is unsatisfiable by evidence due to hard constraint violation {} (see above)'.
                                format(global_bpll_grounding.mrf.formulas[formula.index]))
            if isinstance(gnd, Logic.Equality):
                # if an equality grounding is false in a conjunction, we can
                # stop since the  conjunction cannot be rendered true in any
                # grounding that follows
                if gnd.truth(None) == 0:
                    continue
                for gf in self._iter_groundings_fast(formula, constituents, c_index + 1, dict_union(assignment, varass),
                                                     variables, false_var, level + 1):
                    yield gf
            else:
                var = self.mrf.variable(gnd.ground_atom)
                world_ = list(self.mrf.evidence)
                stat = []
                skip = False
                false_var_ = false_var
                vars_ = list(variables)
                for val_index, value in var.iter_values():
                    var.setval(value, world_)
                    truth = gnd(world_)
                    if truth == 0 and value == var.evidence_value():
                        # if the evidence value renders the current
                        # constituent false and there was already a false
                        # literal in the grounding path, we can prune the
                        # tree since no grounding will be true
                        if false_var is not None and false_var != var.index:
                            skip = True
                            break
                        else:
                            # if there was no literal false so far, we collect statistics only for the current literal
                            # and only if all future literals will be true by evidence
                            vars_ = []
                            false_var_ = var.index
                    if truth > 0 and false_var is None:
                        stat.append((var.index, val_index, truth))
                if false_var is not None and false_var == var.index:
                    # in case of non-mutual exclusive values take only the values that render all literals true
                    # example: soft-functional constraint with !foo(?x) ^ foo(?y), x={X,Y,Z} where the evidence
                    # foo(Z) is true, here the grounding !foo(X) ^ foo(Y) is false:
                    # !foo(X) is true for foo(Z) and foo(Y) and
                    # (!foo(Z) ^ !foox(X) ^ !foo(Y)), foo(Y) is true for foo(Y), both are only true for foo(Y)
                    stat = set(variables).intersection(stat)
                    skip = not bool(stat)  # skip if no values remain
                if skip:
                    continue
                for gf in self._iter_groundings_fast(formula, constituents, c_index + 1, dict_union(assignment, varass),
                                                     vars_ + stat, false_var=false_var_, level=level + 1):
                    yield gf

    def _iter_groundings(self, simplify=False, unsatfailure=False):
        global global_bpll_grounding
        global_bpll_grounding = self
        # pdb.set_trace()
        if self.multicore:
            pool = Pool(maxtasksperchild=1)
            i = 0
            try:
                for ground_result in pool.imap(with_tracing(create_formula_groundings), self.formulas):
                    if self.verbose:
                        bar = ProgressBar(color='green')
                        bar.update(float(i + 1) / float(len(self.formulas)))
                        i += 1

                    for f_index, stat in ground_result:
                        for (var_index, val_index, val) in stat:
                            self._var_index2f_index[var_index].add(f_index)
                            self._addstat(f_index, var_index, val_index, val)
                        check_mem()
                    yield None
            except Exception as e:
                logger.error('Error in child process. Terminating pool...')
                pool.close()
                raise e
            finally:
                pool.terminate()
                pool.join()
        else:
            for ground_result in map(create_formula_groundings, self.formulas):
                for f_index, stat in ground_result:
                    for (var_index, val_index, val) in stat:
                        self._var_index2f_index[var_index].add(f_index)
                        self._addstat(f_index, var_index, val_index, val)
                yield None

    def _addstat(self, f_index, var_index, val_index, inc=1):
        if f_index not in self._stat:
            self._stat[f_index] = {}
        d = self._stat[f_index]
        if var_index not in d:
            d[var_index] = [0] * self.mrf.variable(var_index).value_count()
        d[var_index][val_index] += inc
