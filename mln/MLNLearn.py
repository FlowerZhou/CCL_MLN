"""
MLN learn
"""
import argparse
import io
from tkinter import *
import sys
import traceback
from dnutils import out, ifnone, logs
from mln.markov_logic_network import parse_mln, MarkovLogicNetwork
from mln.project import MLNProject, MLNConfig
import tkinter.messagebox
import fnmatch
from mln.method import LearningMethods
from cProfile import Profile

from mln.util import headline, StopWatch
from tkinter.filedialog import asksaveasfilename, askopenfilename

from tabulate import tabulate
import pstats
from mln.database import DataBase, parse_db
from mln.learning.common import DiscriminativeLearner
import os
from mln.plogic import PLogicFramework

logger = logs.getlogger(__name__)

QUERY_PREDS = 0
EVIDENCE_PREDS = 1


class MLNLearn(object):
    """
    Wrapper class for learning using a MLN configuration.

    :param config: Instance of a :class:`mln.MLNConfig` class
                   representing a serialized configuration. Any parameter
                   in the config object can be overwritten by a respective
                   entry in the ``params`` dict.

    :example:

        >>> conf = MLNConfig('path/to/config/file')
        # overrides the MLN and database to be used.
        >>> learn = MLNLearn(conf, mln=newmln, db=newdb)

    .. seealso::
        :class:`mln.MLNConfig`

    """

    def __init__(self, config=None, **params):
        self.configfile = None
        if config is None:
            self._config = {}
        elif isinstance(config, MLNConfig):
            self._config = config.config
            self.configfile = config
        self._config.update(params)

    @property
    def mln(self):
        """
        The :class:`mln.MLN` instance to be used for learning.
        """
        return self._config.get('mln')

    @property
    def db(self):
        """
        The :class:`mln.Database` instance to be used for learning.
        """
        return self._config.get('db')

    @property
    def output_filename(self):
        """
        The name of the file the learnt MLN is to be saved to.
        """
        return self._config.get('output_filename')

    @property
    def params(self):
        """
        A dictionary of additional parameters that are specific to a
        particular learning algorithm.
        """
        return eval("dict(%s)" % self._config.get('params', ''))

    @property
    def method(self):
        """
        The string identifier of the learning method to use. Defaults to
        ``'BPLL'``.
        """
        return LearningMethods.clazz(self._config.get('method', 'BPLL'))

    @property
    def pattern(self):
        """
        A Unix file pattern determining the database files for learning.
        """
        return self._config.get('pattern', '')

    @property
    def use_prior(self):
        """
        Boolean specifying whether or not to use a prio distribution for
        parameter learning. Defaults to ``False``
        """
        return self._config.get('use_prior', False)

    @property
    def prior_mean(self):
        """
        The mean of the gaussian prior on the weights. Defaults to ``0.0``.
        """
        return float(self._config.get('prior_mean', 0.0))

    @property
    def prior_stdev(self):
        """
        The standard deviation of the prior on the weights. Defaults to
        ``5.0``.
        """
        return float(self._config.get('prior_stdev', 5.0))

    @property
    def incremental(self):
        """
        Specifies whether or incremental learning shall be enabled.
        Defaults to ``False``.

        .. note::
            This parameter is currently unused.

        """
        return self._config.get('incremental', False)

    @property
    def shuffle(self):
        """
        Specifies whether or not learning databases shall be shuffled before
        learning.

        .. note::
            This parameter is currently unused.
        """
        self._config.get('shuffle', False)
        return True

    @property
    def use_initial_weights(self):
        """
        Specifies whether or not the weights of the formulas prior to learning
        shall be used as an initial guess for the optimizer. Default is
        ``False``.
        """
        return self._config.get('use_initial_weights', True)

    @property
    def qpreds(self):
        """
        A list of predicate names specifying the query predicates in
        discriminative learning.

        .. note::
            This parameters only affects discriminative learning methods and
            is mutually exclusive with the :attr:`mln.MLNLearn.epreds`
            parameter.
        """
        return self._config.get('qpreds', '').split(',')

    @property
    def epreds(self):
        """
        A list of predicate names specifying the evidence predicates in
        discriminative learning.

        .. note::
            This parameters only affects discriminative learning methods and
            is mutually exclusive with the :attr:`mln.MLNLearn.qpreds`
            parameter.
        """
        return self._config.get('epreds', '').split(',')

    @property
    def discr_preds(self):
        """
        Specifies whether the query predicates or the evidence predicates
        shall be used. In either case, the respective other case will be
        automatically determined, i.e. if a list of query predicates is
        specified and ``disc_preds`` is ``mln.QUERY_PREDS``, then all
        other predicates will represent the evidence predicates and vice
        versa. Possible values are ``mln.QUERY_PREDS`` and
        ``mln.EVIDENCE_PREDS``.
        """
        return self._config.get('discr_preds', QUERY_PREDS)

    @property
    def logic(self):
        """
        String identifying the logical calculus to be used in the MLN. Must be
        either ``'FirstOrderLogic'``
        or ``'FuzzyLogic'``.

        .. note::
            It is discouraged to use the ``FuzzyLogic`` calculus for learning
            MLNs. Default is ``'FirstOrderLogic'``.
        """
        return self._config.get('logic', 'FirstOrderLogic')

    @property
    def grammar(self):
        """
        String identifying the MLN syntax to be used. Allowed values are
        ``'StandardGrammar'`` and ``'Grammar'``. Default is
        ``'Grammar'``.
        """
        return self._config.get('grammar', 'Grammar')

    @property
    def multicore(self):
        """
        Specifies if all cores of the CPU are to be used for learning.
        Default is ``False``.
        """
        return self._config.get('multicore', False)

    @property
    def profile(self):
        """
        Specifies whether or not the Python profiler shall be used. This is
        convenient for debugging and optimizing your code in case you have
        developed own algorithms. Default is ``False``.
        """
        return self._config.get('profile', False)

    @property
    def verbose(self):
        """
        If ``True``, prints some useful output, status and progress
        information to the console. Default is ``False``.
        """
        return self._config.get('verbose', False)

    @property
    def ignore_unknown_preds(self):
        """
        By default, if an atom occurs in a database that is not declared in
        the attached MLN, `mln` will raise a
        :class:`NoSuchPredicateException`. If ``ignore_unknown_preds`` is
        ``True``, undeclared predicates will just be ignored.
        """
        return self._config.get('ignore_unknown_preds', False)

    @property
    def ignore_zero_weight_formulas(self):
        """
        When formulas in MLNs get more complex, there might be the chance that
        some of the formulas retain a weight of zero (because of strong
        independence assumptions in the Learner, for instance). Since such
        formulas have no effect on the semantics of an MLN but on the runtime
        of inference, they can be omitted in the final learnt MLN by settings
        ``ignore_zero_weight_formulas`` to ``True``.
        """
        return self._config.get('ignore_zero_weight_formulas', False)

    @property
    def save(self):
        """
        Specifies whether or not the learnt MLN shall be saved to a file.

        .. seealso::
            :attr:`mln.MLNLearn.output_filename`
        """
        return self._config.get('save', False)

    def run(self):
        """
        Run the MLN learning with the given parameters.
        """
        # load the MLN
        if isinstance(self.mln, MarkovLogicNetwork):
            mln = self.mln
        else:
            raise Exception('No MLN specified')

        # load the training databases
        if type(self.db) is list and all(
                map(lambda e: isinstance(e, DataBase), self.db)):
            dbs = self.db
        elif isinstance(self.db, DataBase):
            dbs = [self.db]
        elif isinstance(self.db, str):
            db = self.db
            if db is None or not db:
                raise Exception('no trainig data given!')
            dbpaths = [os.path.join(self.directory, 'db', db)]
            dbs = []
            for p in dbpaths:
                dbs.extend(DataBase.load(mln, p, self.ignore_unknown_preds))
        else:
            raise Exception(
                'Unexpected type of training databases: %s' % type(self.db))
        if self.verbose:
            print('loaded %d database(s).' % len(dbs))

        watch = StopWatch()

        if self.verbose:
            confg = dict(self._config)
            confg.update(eval("dict(%s)" % self.params))
            if type(confg.get('db', None)) is list:
                confg['db'] = '%d Databases' % len(confg['db'])
            print(tabulate(
                sorted(list(confg.items()), key=lambda key_v: str(key_v[0])),
                headers=('Parameter:', 'Value:')))

        params = dict([(k, getattr(self, k)) for k in (
            'multicore', 'verbose', 'profile', 'ignore_zero_weight_formulas')])

        # for discriminative learning
        if issubclass(self.method, DiscriminativeLearner):
            if self.discr_preds == QUERY_PREDS:  # use query preds
                params['qpreds'] = self.qpreds
            elif self.discr_preds == EVIDENCE_PREDS:  # use evidence preds
                params['epreds'] = self.epreds

        # gaussian prior settings
        if self.use_prior:
            params['prior_mean'] = self.prior_mean
            params['prior_stdev'] = self.prior_stdev
        # expand the parameters
        params.update(self.params)

        if self.profile:
            prof = Profile()
            print('starting profiler...')
            prof.enable()
        else:
            prof = None
        # set the debug level
        olddebug = logger.level
        logger.level = eval('logs.%s' % params.get('debug', 'WARNING').upper())
        mlnlearnt = None
        try:
            # run the learner
            # mlnlearnt = mln.learn(dbs, self.method, **params)
            mlnlearnt = mln
            if self.verbose:
                print()
                print(headline('LEARNT MARKOV LOGIC NETWORK'))
                print()
                mlnlearnt.write()
        except SystemExit:
            print('Cancelled...')
        finally:
            if self.profile:
                prof.disable()
                print(headline('PROFILER STATISTICS'))
                ps = pstats.Stats(prof, stream=sys.stdout).sort_stats(
                    'cumulative')
                ps.print_stats()
            # reset the debug level
            logger.level = olddebug
        print()
        watch.finish()
        watch.print_steps()
        return mlnlearnt








