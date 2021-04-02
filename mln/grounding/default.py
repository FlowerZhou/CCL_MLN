from dnutils import logs, ProgressBar, ifnone
import pdb
from mln.util import fstr, dict_union, StopWatch
from mln.constants import auto, HARD

logger = logs.getlogger(__name__)
CACHE_SIZE = 100000


class DefaultGroundingFactory:
    """
    Implementation of the default grounding algorithm, which
    creates ALL ground atoms and ALL ground formulas.
    :param simplify:        if `True`, the formula will be simplified according to the
                            evidence given.
    :param unsatfailure:    raises a :class:`mln.errors.SatisfiabilityException` if a
                            hard logical constraint is violated by the evidence.
    """

    def __init__(self, mrf, simplify=False, unsatfailure=False, formulas=None, cache=auto, **params):
        self.mrf = mrf
        self.formulas = ifnone(formulas, list(self.mrf.formulas))
        self.total_gf = 0
        for f in self.formulas:
            self.total_gf += f.count_groundings(self.mrf)
        self.grounder = None
        self._cache_size = CACHE_SIZE if cache is auto else cache
        self._cache = None
        self.__cache_init = False
        self.__cache_complete = False
        self._params = params
        self.watch = StopWatch()
        self.simplify = simplify
        self.unsatfailure = unsatfailure

    @property
    def verbose(self):
        return self._params.get('verbose', False)

    @property
    def multicore(self):
        return self._params.get('multicore', True)

    @property
    def is_cached(self):
        return self._cache is not None and self.__cache_init

    @property
    def use_cache(self):
        return self._cache_size is not None and self._cache_size > 0

    def _cache_init(self):

        self._cache = []
        self.__cache_init = True

    def iter_groundings(self):
        """
        Iterates over all formula groundings.
        """
        self.watch.tag('grounding', verbose=self.verbose)
        if self.grounder is None:
            self.grounder = iter(self._iter_groundings(simplify=self.simplify, unsatfailure=self.unsatfailure))
        if self.use_cache and not self.is_cached:
            self._cache_init()
        counter = -1
        while True:
            counter += 1
            if self.is_cached and len(self._cache) > counter:
                yield self._cache[counter]
            elif not self.__cache_complete:
                try:
                    # pdb.set_trace()
                    gf = next(self.grounder)
                    # gf = self.grounder
                except StopIteration:
                    self.__cache_complete = True
                    return
                else:
                    if self._cache is not None:
                        self._cache.append(gf)
                    yield gf
            else:
                return
        self.watch.finish('grounding')
        # print("counter is %d" % counter)
        if self.verbose:
            print()

    def _iter_groundings(self, simplify=False, unsatfailure=False):
        if self.verbose:
            bar = ProgressBar(color='green')
        for i, formula in enumerate(self.formulas):
            if self.verbose:
                # print("begin iter grounding, grounding number is %f" %float(len(self.formulas)))
                bar.update((i + 1) / float(len(self.formulas)))
            # pdb.set_trace()
            for ground_formula in formula.iter_groundings(self.mrf, simplify=simplify):
                if unsatfailure and ground_formula.weight == HARD and ground_formula(self.mrf.evidence) == 0:
                    # print("here to stay! ")
                    ground_formula.print_structure(self.mrf.evidence)
                    raise Exception(
                        'MLN is unsatisfiable due to hard constraint violation %s (see above)' % self.mrf.formulas[
                            ground_formula.index])
                yield ground_formula


class EqualityConstraintGrounder(object):
    """
    Grounding factory for equality constraints only.
    """

    def __init__(self, mrf, domains, mode, eq_constraints):
        """
        Initialize the equality constraint grounder with the given MLN
        and formula. A formula is required that contains all variables
        in the equalities in order to infer the respective domain names.

        :param mode: either ``alltrue`` or ``allfalse``
        """
        self.constraints = eq_constraints
        self.mrf = mrf
        self.truth = {'alltrue': 1, 'allfalse': 0}[mode]
        self.mode = mode
        eqvars = [c for eq in eq_constraints for c in eq.args if self.mrf.mln.logic.isvar(c)]
        self.vardomains = dict([(v, d) for v, d in domains.items() if v in eqvars])

    def iter_valid_variable_assignments(self):
        """
        Yields all variable assignments for which all equality constraints
        evaluate to true.
        """
        return self._iter_valid_variable_assignments(self.vardomains.keys(), {}, self.constraints)

    def _iter_valid_variable_assignments(self, variables, assignments, eq_groundings):
        if not variables:
            yield assignments
            return
        eq_groundings = [eq for eq in eq_groundings if not all([not self.mrf.mln.logic.isvar(a) for a in eq.args])]
        variable = variables[0]
        for value in self.mrf.domains[self.vardomains[variable]]:
            new_eq_groundings = []
            goon = True
            for eq in eq_groundings:
                geq = eq.ground(None, {variable: value}, partial=True)
                t = geq(None)
                if t is not None and t != self.truth:
                    goon = False
                    break
                new_eq_groundings.append(geq)
            if not goon:
                continue
            for assignment in self._iter_valid_variable_assignments(variables[1:],
                                                                    dict_union(assignments, {variable: value}),
                                                                    new_eq_groundings):
                yield assignment

    @staticmethod
    def vardoms_from_formula(mln, formula, *varnames):
        if isinstance(formula, str):
            formula = mln.logic.parse_formula(formula)
        vardomains = {}
        f_vardomains = formula.vardoms(mln)
        for var in varnames:
            if var not in f_vardomains:
                raise Exception('Variable %s not bound to a domain by formula %s' % (var, fstr(formula)))
            vardomains[var] = f_vardomains[var]
        return vardomains
