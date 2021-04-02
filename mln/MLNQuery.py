"""
MLN Inference
"""

import argparse

from tkinter import *
from tkinter.filedialog import askopenfilename, asksaveasfilename
import os
import ntpath
import tkinter.messagebox
import traceback

from dnutils import logs, ifnone, out
from mln.project import MLNProject, MLNConfig, MLNPath
from mln.method import InferenceMethods
from mln.util import parse_queries, headline, StopWatch
from mln.markov_logic_network import parse_mln, MarkovLogicNetwork
from mln.database import parse_db, DataBase
from tabulate import tabulate
from cProfile import Profile
import pstats
import io
import pdb
from mln.em_learn import EMFramework
from mln.plogic import PLogicFramework


logger = logs.getlogger(__name__)


SETTINGS = ['db', 'method', 'output_filename', 'save', 'grammar', 'queries']


class MLNQuery(object):

    def __init__(self, config=None, verbose=None, **params):
        """
        Class for performing MLN inference
        :param config:  the configuration file for the inference
        :param verbose: boolean value whether verbosity logs will be
                        printed or not
        :param params:  dictionary of additional settings
        """
        self.configfile = None
        if config is None:
            self._config = {}
        elif isinstance(config, MLNConfig):
            self._config = config.config
            self.configfile = config
        if verbose is not None:
            self._verbose = verbose
        else:
            self._verbose = self._config.get('verbose', False)
        self._config.update(params)

    @property
    def mln(self):
        return self._config.get('mln')

    @property
    def db(self):
        return self._config.get('db')

    @property
    def output_filename(self):
        return self._config.get('output_filename')

    @property
    def params(self):
        return eval("dict(%s)" % self._config.get('params', ''))

    @property
    def method(self):
        return InferenceMethods.clazz(self._config.get('method', 'MC-SAT'))

    @property
    def queries(self):
        q = self._config.get('queries', ALL)
        if isinstance(q, str):
            return parse_queries(self.mln, q)
        return q


    @property
    def cw(self):
        return self._config.get('cw', False)

    @property
    def cw_preds(self):
        preds = self._config.get('cw_preds', '')
        if type(preds) is str:
            preds = preds.split(',')
        return map(str.strip, preds)

    @property
    def logic(self):
        return self._config.get('logic', 'FirstOrderLogic')

    @property
    def grammar(self):
        return self._config.get('grammar', 'Grammar')

    @property
    def multicore(self):
        return self._config.get('multicore', False)

    @property
    def profile(self):
        return self._config.get('profile', False)

    @property
    def verbose(self):
        return self._verbose

    @property
    def ignore_unknown_preds(self):
        return self._config.get('ignore_unknown_preds', False)

    @property
    def save(self):
        return self._config.get('save', False)

    def run(self):
        watch = StopWatch()
        watch.tag('inference', self.verbose)
        # load the MLN
        if isinstance(self.mln, MarkovLogicNetwork):
            mln = self.mln
        else:
            raise Exception('No MLN specified')
        # load the database
        if isinstance(self.db, DataBase):
            db = self.db
        elif isinstance(self.db, list) and len(self.db) == 1:
            db = self.db[0]
        elif isinstance(self.db, list) and len(self.db) == 0:
            db = DataBase(mln)
        elif isinstance(self.db, list):
            raise Exception(
                'Got {} dbs. Can only handle one for inference.'.format(
                    len(self.db)))
        else:
            raise Exception('DB of invalid format {}'.format(type(self.db)))

        # expand the parameters
        params = dict(self._config)
        if 'params' in params:
            params.update(eval("dict(%s)" % params['params']))
            del params['params']
        params['verbose'] = self.verbose
        if self.verbose:
            print(tabulate(sorted(list(params.items()), key=lambda k_v: str(k_v[0])), headers=('Parameter:', 'Value:')))
        if type(db) is list and len(db) > 1:
            raise Exception('Inference can only handle one database at a time')
        elif type(db) is list:
            db = db[0]
        params['cw_preds'] = filter(lambda x: bool(x), self.cw_preds)
        # extract and remove all non-algorithm

        for s in SETTINGS:
            if s in params:
                del params[s]

        if self.profile:
            prof = Profile()
            print('starting profiler...')
            prof.enable()
        # set the debug level
        olddebug = logger.level
        logger.level = eval('logs.%s' % params.get('debug', 'WARNING').upper())
        result = None
        # add according to ExpressGNN
        # pdb.set_trace()
        # em = EMFramework(mln, db)
        for q in self.queries:
            val = 0 if q[0] == '!' else 1
            q = mln.logic.parse_formula(q)
            db.test_fact_ls.append((val, q.pred_name, tuple(q.args)))
            db.test_fact_dict[q.pred_name].add((val, tuple(q.args)))
            db.add_ht(q.pred_name, q.args, db.ht_dict, 1)

        try:
            # pdb.set_trace()
            # em = EMFramework(mln, db)
            # em.em_procedure()
            plogic = PLogicFramework(mln, db)
            plogic.kge_process("TransE")
            print(">>>>>>>>EM finished......")
            pdb.set_trace()
            mln_ = mln.materialize(db)
            mrf = mln_.ground(db)
            # pdb.set_trace()
            inference = self.method(mrf, self.queries, **params)
            if self.verbose:
                print()
                print(headline('EVIDENCE VARIABLES'))
                print()
                # mrf.print_evidence_vars()

            result = inference.run()
            if self.verbose:
                print()
                print(headline('INFERENCE RESULTS'))
                print()
                # pdb.set_trace()
                inference.write()
            if self.verbose:
                print()
                inference.write_elapsed_time()
        except SystemExit:
            traceback.print_exc()
            print('Cancelled...')
        finally:
            if self.profile:
                prof.disable()
                print(headline('PROFILER STATISTICS'))
                ps = pstats.Stats(prof, stream=sys.stdout).sort_stats('cumulative')
                ps.print_stats()
            # reset the debug level
            logger.level = olddebug
        if self.verbose:
            print()
            watch.finish()
            watch.print_steps()
        return result
