import os
import pdb
import sys
sys.path.append(os.getcwd())
print(sys.path)
from mln.project import MLNConfig
import os
from mln.MLNLearn import MLNLearn
from mln.markov_logic_network import MarkovLogicNetwork
from mln.database import DataBase
from mln.MLNQuery import MLNQuery
from utils import config, locs
from utils.config import global_config_filename
from mln.util import elapsed_time_str
import time
import pytest


class SocialModelling(object):

    def __init__(self):
        pass

    def read_data(self, paths):
        # pdb.set_trace()
        content = []
        base_path = os.getcwd()
        file_ = open(base_path + '/' + paths, 'r')
        pre_content = file_.read()
        pre_content = pre_content.split('###')
        pre_content = [x for x in pre_content if x != '']
        for i in pre_content:
            element = i.split('\n')
            element = [x.replace(':', '_') for x in element if x != '']
            # for j in element[1::]:
                # splited = j.split('(')
                # content.append((element[0], splited[0]+'('+splited[1].upper()))
            for j in element:
                content.append(j)

        return content

    def read_formula(self, paths):
        formula = []
        base_path = os.getcwd()
        file = open(base_path + '/' + paths, 'r')
        formula = file.read()
        formula = formula.split('\n')
        formula = [x for x in formula if x != '']
        formula = [' ' + x.replace(' or ', ' v ').replace(' and ', ' ^ ').replace(':', '') for x in formula]
        exist_list = [x for x in formula if 'Exists ' in x]
        formula = [x for x in formula if 'Exists ' not in x]
        return formula

    def read_predicate(self, paths):
        predicate = []
        base_path = os.getcwd()
        file_ = open(base_path + '/' + paths, 'r')
        predicate = file_.read()
        predicate = predicate.split('\n')
        # predicate.remove('')
        predicate = [pred for pred in predicate if pred != '']
        predicate_list = [x.split('(')[0] for x in predicate]
        predicate_list2 = [x.split('(')[1].replace(' ', '').lower() for x in predicate]
        # predicate_ = []
        for i in zip(predicate_list, predicate_list2):
        # for i in predicate:
            predicate.append(i[0] + '(' + i[1])
        # predicate_.append(i)
        return predicate

    def model_config(self, predicate, formula, database, mln_path, db_path):
        base_path = os.getcwd()
        mln = MarkovLogicNetwork(grammar='StandardGrammar', logic='FirstOrderLogic')
        for i in predicate:
            mln << i
            print('input predicate successful:'+i)
        # pdb.set_trace()
        for i in formula:
            mln << i
            print('input formula successful :'+i)
        mln.write()
        mln.to_file(base_path + '/' + mln_path)
        db = DataBase(mln)
        # pdb.set_trace()

        for _, d in enumerate(database):
            db << d
            print('input database successful : ' + d)
        db.write()
        db.to_file(base_path + '/' + db_path)
        return db, mln

    def activate_model(self, database, mln):

        DEFAULT_CONFIG = os.path.join(locs.user_data, global_config_filename)
        conf = MLNConfig(DEFAULT_CONFIG)
        config = {}
        config['verbose'] = True
        config['discr_preds'] = 0
        config['db'] = database
        config['mln'] = mln
        config['ignore_zero_weight_formulas'] = 1    # 0
        config['ignore_unknown_preds'] = True   # 0
        config['grammar'] = 'StandardGrammar'
        config['logic'] = 'FirstOrderLogic'
        config['method'] = 'BPLL_CG'    # Other method: 'BPLL_CG', 'CLL'
        config['multicore'] = True
        config['profile'] = 0
        config['shuffle'] = 0
        config['prior_mean'] = 0
        config['prior_stdev'] = 10   # 5
        config['save'] = True
        config['use_initial_weights'] = True
        config['use_prior'] = 0
        # config['output_filename'] = 'learnt.dbpll_cg.student-new-train-student-new-2.mln'

        config['infoInterval'] = 500
        config['resultsInterval'] = 1000
        conf.update(config)

        print('training...')
        learn = MLNLearn(conf, mln=mln, db=database)
        # learn.output_filename(r'C:\Users\anaconda3 4.2.0\test.mln')
        start = time.time()
        result = learn.run()
        end = time.time()
        print('finished...')
        print('Learning took %s ' % elapsed_time_str(end - start))

        return result

    def inference(self, path, result, data, mln):
        query_list = []
        base_path = os.getcwd()
        file = open(base_path + '/' + path, 'r')
        query_list = file.read()
        query_list = query_list.split('\n')
        query_list = [x for x in query_list if x != '']
        for i in query_list:
            print(MLNQuery(queries=i, method='GibbsSampler', mln=mln, db=data, verbose=False, multicore=True).
                  run().results)
        # Other Methods: EnumerationAsk, MC-SAT, WCSPInference, GibbsSampler

    def inference_str(self, string, result, data, mln):
        print(MLNQuery(queries=string, method='MCSAT', mln=mln, db=data, verbose=True, multicore=True,
                       save=True, output_filename=r'learnt.dbpll_cg.student-new-train-student-new-2.mln').run().results)


if __name__ == '__main__':

    sm = SocialModelling()

    predicate = sm.read_predicate('predicate.txt')
    formula = sm.read_formula('formula1.txt')
    database = sm.read_data('data.txt')
    # pdb.set_trace()
    data, mln = sm.model_config(predicate, formula, database, 'smoke.mln', 'smoke.db')
    # mln = MarkovLogicNet(grammar='StandardGrammar', logic='FirstOrderLogic')
    # mln_ = mln.load('uw.mln')
    # db = DataBase(mln_)
    # data = db.load('language.db')
    # pdb.set_trace()
    output = sm.activate_model(data, mln)
    output.to_file(os.getcwd() + '/' + 'learnt_smoke_mln.mln')
    # pdb.set_trace()
    # sm.inference_str('advisedBy(Person18,Person248)', output, data, mln)
    sm.inference_str('Smokes(Nixon)', output, data, mln)
    # inference = social_modelling.inference('inference.txt', output, data, mln)
    """
    output = ""
    mln = MarkovLogicNetwork(grammar='StandardGrammar', logic='FirstOrderLogic')
    #
    pdb.set_trace()
    mln_ = mln.load('smokes')
    db = DataBase(mln_)
    data = db.load(mln_, 'smokes.db')
    mln_.materialize(db)
    mln_.ground(db)
    sm.inference_str('Smokes(Nixon)', output, data, mln_)
    """







