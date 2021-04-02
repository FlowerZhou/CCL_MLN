from dnutils import logs, out
from dnutils.console import barstr

from logic.elements import Logic
from mln.database import DataBase
from mln.constants import ALL
from mln.mrfvars import MutexVariable, SoftMutexVariable, FuzzyVariable
from mln.util import (StopWatch, elapsed_time_str, headline, tty, edict)
import sys
from mln.mlnpreds import SoftFunctionalPredicate, FunctionalPredicate
import pdb
logger = logs.getlogger(__name__)


class Inference(object):
    """
    Represents a super class for all inference methods.
    Also provides some convenience methods for collecting statistics
    about the inference process and nicely outputting results.

    :param mrf:        the MRF inference is being applied to.
    :param queries:    a query or list of queries, can be either instances of
                       :class:`pracmln.logic.common.Logic` or string representations of them,
                       or predicate names that get expanded to all of their ground atoms.
                       If `ALL`, all ground atoms are subject to inference.

    Additional keyword parameters:

    :param cw:         (bool) if `True`, the closed-world assumption will be applied
                       to all but the query atoms.
    """

    def __init__(self, mrf, queries=ALL, **params):
        # pdb.set_trace()
        self.mrf = mrf
        self.mln = mrf.mln
        self._params = edict(params)
        if not queries:
            self.queries = [self.mln.logic.ground_literal(ga, negated=False, mln=self.mln) for ga in
                            self.mrf.ground_atom if self.mrf.evidence[ga.index] is None]
        else:
            # check for single/multiple query and expand
            # pdb.set_trace()
            if type(queries) is not list:
                queries = [queries]
            self.queries = self._expand_queries(queries)
        # fill in the missing truth values of variables that have only one remaining value
        for variable in self.mrf.variables:
            if variable.value_count(self.mrf.evidence_dicti()) == 1:  # the var is fully determined by the evidence
                for _, value in variable.iter_values(self.mrf.evidence):
                    break
                self.mrf.set_evidence(variable.value2dict(value), erase=False)
        # apply the closed world assumptions to the explicitly specified predicates
        if self.cwpreds:
            for pred in self.cwpreds:
                if isinstance(self.mln.predicate(pred), SoftFunctionalPredicate):
                    if self.verbose:
                        logger.warning('Closed world assumption will be applied to soft functional predicate %s' % pred)
                elif isinstance(self.mln.predicate(pred), FunctionalPredicate):
                    raise Exception('Closed world assumption is inapplicable to functional predicate %s' % pred)
                for ground_atom in self.mrf.ground_atoms:
                    if ground_atom.pred_name != pred:
                        continue
                    if self.mrf.evidence[ground_atom.index] is None:
                        self.mrf.evidence[ground_atom.index] = 0
        # apply the closed world assumption to all remaining ground atoms that are not in the queries
        if self.closedworld:
            qpreds = set()
            for q in self.queries:
                qpreds.update(q.pred_names())
            for ground_atom in self.mrf.ground_atoms:
                if isinstance(self.mln.predicate(ground_atom.pred_name), FunctionalPredicate) \
                        and not isinstance(self.mln.predicate(ground_atom.pred_name), SoftFunctionalPredicate):
                    continue
                if ground_atom.pred_name not in qpreds and self.mrf.evidence[ground_atom.index] is None:
                    self.mrf.evidence[ground_atom.index] = 0
        for var in self.mrf.variables:
            if isinstance(var, FuzzyVariable):
                var.consistent(self.mrf.evidence, strict=True)
        self._watch = StopWatch()

    @property
    def verbose(self):
        return self._params.get('verbose', False)

    @property
    def results(self):
        if self._results is None:
            raise Exception('No results available. Run the inference first.')
        else:
            return self._results

    @property
    def elapsed_time(self):
        return self._watch['inference'].elapsed_time

    @property
    def multicore(self):
        return self._params.get('multicore')

    @property
    def resultdb(self):
        if '_resultdb' in self.__dict__:
            return self._resultdb
        db = DataBase(self.mrf.mln)
        for atom in sorted(self.results, key=str):
            db[str(atom)] = self.results[atom]
        return db

    @property
    def closedworld(self):
        return self._params.get('cw', False)

    @property
    def cwpreds(self):
        return self._params.get('cw_preds', [])

    def _expand_queries(self, queries):
        """
        Expands the list of queries where necessary, e.g. queries that are
        just predicate names are expanded to the corresponding list of atoms.
        """
        equeries = []
        # pdb.set_trace()
        for query in queries:
            if type(query) == str:
                prevLen = len(equeries)
                if '(' in query:  # a fully or partially grounded formula
                    f = self.mln.logic.parse_formula(query)
                    for gf in f.iter_groundings(self.mrf):
                        equeries.append(gf)
                else:  # just a predicate name
                    if query not in self.mln.pred_names:
                        raise Exception('Unsupported query: %s is not among the admissible predicates.' % query)
                        continue
                    for ground_atom in self.mln.predicate(query).ground_atoms(self.mln, self.mrf.domains):
                        equeries.append(self.mln.logic.ground_literal(self.mrf.ground_atom(ground_atom), negated=False,
                                                                      mln=self.mln))
                if len(equeries) - prevLen == 0:
                    raise Exception("String query '%s' could not be expanded." % query)
            elif isinstance(query, Logic.Formula):
                equeries.append(query)
            else:
                raise Exception("Received query of unsupported type '%s'" % str(type(query)))
        return equeries

    def _run(self):
        raise Exception('%s does not implement _run()' % self.__class__.__name__)

    def run(self):
        """
        Starts the inference process.
        """

        # perform actual inference (polymorphic)
        if self.verbose:
            print('Inference engine: %s' % self.__class__.__name__)
        self._watch.tag('inference', verbose=self.verbose)
        # pdb.set_trace()
        # self.mrf.reduce_variables(self.mrf.variables, self.queries)
        _weights_backup = list(self.mln.weights)
        self._results = self._run()
        self.mln.weights = _weights_backup
        self._watch.finish('inference')
        return self

    def write(self, stream=sys.stdout, color=None, sort='prob', group=True, reverse=True):
        barwidth = 30
        if tty(stream) and color is None:
            color = 'yellow'
        if sort not in ('alpha', 'prob'):
            raise Exception('Unknown sorting: %s' % sort)
        results = dict(self.results)
        if group:
            for var in sorted(self.mrf.variables, key=str):
                res = dict([(atom, prob) for atom, prob in results.items() if atom in map(str, var.ground_atoms)])
                if not res:
                    continue
                if isinstance(var, MutexVariable) or isinstance(var, SoftMutexVariable):
                    stream.write('%s:\n' % var)
                if sort == 'prob':
                    res = sorted(res, key=self.results.__getitem__, reverse=reverse)
                elif sort == 'alpha':
                    res = sorted(res, key=str)
                for atom in res:
                    stream.write('%s %s\n' % (barstr(barwidth, self.results[atom], color=color), atom))
            return
        # first sort wrt to probability
        results = sorted(results, key=self.results.__getitem__, reverse=reverse)
        # then wrt gnd atoms
        results = sorted(results, key=str)
        for q in results:
            stream.write('%s %s\n' % (barstr(barwidth, self.results[q], color=color), q))
        self._watch.print_steps()

    def write_elapsed_time(self, stream=sys.stdout, color=None):
        if stream is sys.stdout and color is None:
            color = True
        elif color is None:
            color = False
        if color:
            col = 'blue'
        else:
            col = None
        total = float(self._watch['inference'].elapsed_time)
        stream.write(headline('INFERENCE RUNTIME STATISTICS'))
        print
        self._watch.finish()
        for t in sorted(self._watch.tags.values(), key=lambda t: t.elapsed_time, reverse=True):
            stream.write('%s %s %s\n' % (
            barstr(width=30, percent=t.elapsed_time / total, color=col), elapsed_time_str(t.elapsed_time), t.label))
