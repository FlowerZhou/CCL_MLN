"""
PLogic Framework for MLN,
according to paper "Probabilistic Logic Neural Networks for Reasoning"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import datetime
from mln.util import augment_triplet, evaluate
from model.model import KGEModel
import pdb
from numpy.ma.core import exp
from math import sqrt
import argparse
import json
import logging
from model.dataloader import TrainDataset, BidirectionalOneShotIterator


iterations = 2
kge_model = 'TransE'
kge_batch = 1024
kge_neg = 256
kge_dim = 100
kge_gamma = 24
kge_alpha = 1
kge_lr = 0.001
kge_iters = 10000
kge_tbatch = 16
kge_reg = 0.0
kge_topk = 10

mln_iters = 100
mln_lr = 0.0001
mln_threads = 8

max_steps = 150

train_triples = []
observed_triplets = []
hidden_triples = []
hidden_triplets = []
test_triples = []
all_triples = []
all_true_triples = []
triplet2prob = dict()
triplets = []
rules_grad = []

nentity = 0
nrelation = 0
triplets_size = 0
observed_triplets_size = 0
hidden_triplets_size = 0


class Triplet(object):
    """
    triplets, correspond to predicates, entity(constant) map to int id
    """
    def __init__(self, h, r, t):
        self.h = h
        self.r = r
        self.t = t
        self.type = -1
        self.valid = -1
        self.truth = 0
        self.logit = 0
        self.rule_ids = []

    def __eq__(self, other):
        return (self.h, self.r, self.t) == (other.h, other.r, other.t)

    def __hash__(self):  # hashable
        return hash((self.h, self.r, self.t))

    def __ne__(self, other):
        return not(self == other)


class Args(object):
    """
    Args for Training and Testing Knowledge Graph Embedding Models
    """
    def __init__(self):
        self.cuda = False
        self.negative_adversarial_sampling = True
        self.uni_weight = False
        self.regularization = 0.0
        self.countries = False
        self.nentity = nentity
        self.nrelation = nrelation
        self.test_batch_size = 16
        self.cpu_num = 10
        self.topk = 100
        self.test_log_steps = 1000
        self.log_steps = 100
        self.do_test = True
        self.record = True
        self.evaluate_train = True
        self.adversarial_temperature = 1.0
        self.save_checkpoint_steps = 1000
        self.save_path = os.getcwd() + '/result/'
        self.batch_size = kge_batch


class PLogicFramework(object):
    """
    PLogic Framework
    """
    def __init__(self, mln, db):
        self.mln = mln
        self.db = db
        self.predicate_dict = dict()
        self.db.data_process()
        self.id2entity = dict()
        self.id2relation = dict()
        for k in self.db.const2ind.keys():
            self.id2entity[self.db.const2ind[k]] = k

    def map_predicate_to_id(self):
        i = 0
        for pred in self.mln.predicate_name:
            if self.predicate_dict.get(pred) is None:
                self.predicate_dict[pred] = i
                self.id2relation[i] = pred
                i += 1

    def sigmoid(self, x):
        return 1.0 / (1.0 + exp(-x))

    def log_metrics(self, mode, step, metrics):
        """
        Print the evaluation logs
        """
        for metric in metrics:
            logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

    def set_logger(self, args):
        """
        set format for logging
        """
        logging.basicConfig()
        log_file = os.path.join(args.save_path, "train.log")
        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=log_file,
            filemode='w'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    def read_triples(self):
        """
        read triples and map them into IDs
        """
        # pdb.set_trace()
        triplets = []
        for pred in self.mln.predicates:
            relation2id = self.predicate_dict[pred.name]
            for arg1 in self.db.const_sort_dict[pred.arg_doms[0]]:
                entity2id1 = self.db.const2ind[arg1]
                for arg2 in self.db.const_sort_dict[pred.arg_doms[1]]:
                    entity2id2 = self.db.const2ind[arg2]
                    if (entity2id1, relation2id, entity2id2) not in triplets:
                        triplets.append((entity2id1, relation2id, entity2id2))
                        all_triples.append((entity2id1, relation2id, entity2id2))
        return triplets

    def get_observed_triplets(self):
        """
        get observed triplets from database
        """
        global observed_triplets_size
        for rel in sorted(self.db.fact_dict.keys()):
            for fact in sorted(list(self.db.fact_dict[rel])):
                val, args = fact
                rel2id = self.predicate_dict[rel]
                ent2id1 = self.db.const2ind[args[0]]
                ent2id2 = self.db.const2ind[args[1]]
                triplet = Triplet(ent2id1, rel2id, ent2id2)
                triplet.valid = 1
                triplet.type = 'o'
                train_triples.append((ent2id1, rel2id, ent2id2))
                observed_triplets.append(triplet)

        observed_triplets_size = len(observed_triplets)

    def get_test_triples(self):
        """
        get test triples (queries), and hidden triplets(is different from pLogic)
        pLogic's hidden triplets are got from rules
        here we simply use |E| × |R| × |E| − |O| without leveraging rule
        """
        global hidden_triples
        global triplets_size
        global all_true_triples
        for _, pred, args in sorted(self.db.test_fact_ls):
            rel2id = self.predicate_dict[pred]
            ent2id1 = self.db.const2ind[args[0]]
            ent2id2 = self.db.const2ind[args[1]]
            if (ent2id1, rel2id, ent2id2) not in test_triples:
                test_triples.append((ent2id1, rel2id, ent2id2))

        hidden_triples = list(set(all_triples) - set(train_triples))
        triplets_size = len(hidden_triples) + observed_triplets_size
        all_true_triples = train_triples + test_triples
        for h, r, t in hidden_triples:
            triplet = Triplet(h, r, t)
            triplet.type = 'h'
            triplet.valid = 0
            hidden_triplets.append(triplet)

    def get_hidden_triplets(self):
        pass

    def get_triplets(self):
        """
        get triplets(observed + hidden)
        """
        for triplet in observed_triplets:
            triplets.append(triplet)
        for triplet in hidden_triplets:
            triplets.append(triplet)

    def link_rule(self):
        """
        link rule to triplets, now has some questions, need to be modified
        """
        # pdb.set_trace()
        for t in triplets:
            pred = self.id2relation[t.r]
            for index, rule in self.mln.iter_formulas():
                for atom in rule.children:
                    if pred == atom.pred_name:
                        t.rule_ids.append(index)
                        break

    def read_probability_of_hidden_triplets(self, file_path, triplet_threshold):
        """
        read annotation of hidden triplets from KGE model, one triplet per line, with the format: <h> <r> <t> <prob>
        """
        # pdb.set_trace()
        with open(file_path, 'r') as f1:
            for line in f1.readlines():
                h_, r_, t_, prob = line.split('\t')
                h = self.db.const2ind[h_]
                r = self.predicate_dict[r_]
                t = self.db.const2ind[t_]
                triplet = Triplet(h, r, t)
                triplet2prob[triplet] = float(prob)
        for i in range(triplets_size):
            if triplets[i].type == 'o':
                triplets[i].truth = 1
                triplets[i].valid = 1
                continue
            if (triplets[i] in triplet2prob.keys()) and (triplet2prob[triplets[i]] >= triplet_threshold):
                triplets[i].truth = triplet2prob[triplets[i]]
                triplets[i].valid = 1
            elif (triplets[i] in triplet2prob.keys()) and (triplet2prob[triplets[i]] < triplet_threshold):
                triplets[i].truth = triplet2prob[triplets[i]]
                triplets[i].valid = 0

        self.link_rule()
        for k in range(len(self.mln.formulas)):
            rules_grad.append(0)
        for i in range(mln_iters):
            error = self.train_epoch(mln_lr)
            # pdb.set_trace()
            print("Iteration : %d %f " % (i, error))

    def train_epoch(self, lr):
        """
        mln training process
        """
        error = 0.0
        cn = 0.0
        for k in range(len(self.mln.formulas)):
            rules_grad[k] = 0
        for i in range(triplets_size):
            rule_num = len(triplets[i].rule_ids)
            if rule_num == 0:
                continue
            triplets[i].logit = 0
            for k in range(rule_num):
                rule_id = triplets[i].rule_ids[k]
                triplets[i].logit += self.mln.formulas[rule_id].weight / rule_num
            triplets[i].logit = self.sigmoid(triplets[i].logit)
            for k in range(rule_num):
                rule_id = triplets[i].rule_ids[k]
                rules_grad[rule_id] += (triplets[i].truth - triplets[i].logit)

            error += (triplets[i].truth - triplets[i].logit) * (triplets[i].truth - triplets[i].logit)
            cn += 1

        for i in range(len(self.mln.formulas)):
            self.mln.formulas[i].weight += lr * rules_grad[i]

        return sqrt(error / cn)

    def output_prediction(self, pred_file):
        """
        output the prediction to file
        """
        # pdb.set_trace()
        with open(pred_file, 'a+') as f1:
            f1.seek(0)
            for line in f1.readlines():
                h_, r_, t_, f_, rk_ = line.split('\t')
                h = self.db.const2ind[h_]
                r = self.predicate_dict[r_]
                t = self.db.const2ind[t_]
                triplet = Triplet(h, r, t)
                f1.write('{}\t{}\t{}\t{}\n'.format(h_, r_, t_, triplet2prob[triplet]))
                for t in triplets:
                    if t == triplet:
                        f1.write('{}\t{}\t{}\t{}\n'.format(h_, r_, t_, t.logit))
                        break
                f1.write('\n')

    def kge_process(self, model):
        """
        use KGE model to train the observed triples
        """
        # pdb.set_trace()
        self.map_predicate_to_id()
        self.read_triples()
        self.get_observed_triplets()
        self.get_test_triples()
        self.get_triplets()
        global nentity, nrelation
        nentity = len(self.db.const2ind)
        nrelation = len(self.mln.predicates)
        args = Args()
        self.set_logger(args)

        train_triplets = train_triples

        for k in range(iterations):
            if model == 'TransE':
                kge_model = KGEModel(
                    model_name='TransE',
                    nentity=nentity,
                    nrelation=nrelation,
                    hidden_dim=kge_dim,
                    gamma=kge_gamma,
                    double_entity_embedding=False,
                    double_relation_embedding=False
                )
                train_data_loader_head = DataLoader(
                    TrainDataset(train_triplets, nentity, nrelation, kge_neg, 'head-batch'),
                    batch_size=kge_batch,
                    shuffle=True,
                    num_workers=max(1, 10//2),
                    collate_fn=TrainDataset.collate_fn
                )

                train_data_loader_tail = DataLoader(
                    TrainDataset(train_triplets, nentity, nrelation, kge_neg, 'tail-batch'),
                    batch_size=kge_batch,
                    shuffle=True,
                    num_workers=max(1, 10//2),
                    collate_fn=TrainDataset.collate_fn
                )

                train_iterator = BidirectionalOneShotIterator(train_data_loader_head, train_data_loader_tail)

                # set training configuration
                current_learning_rate = kge_lr
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = max_steps // 2
                logging.info('Randomly Initializing %s Model...' % model)
                init_step = 0
                step = init_step
                logging.info('Start Training...')
                logging.info('init_step = %d' % init_step)
                logging.info('learning_rate = %d' % current_learning_rate)
                logging.info('batch_size = %d' % kge_batch)
                logging.info('negative_adversarial_sampling = %d' % True)
                logging.info('hidden_dim = %d' % kge_dim)
                logging.info('gamma = %f' % kge_gamma)
                logging.info('adversarial_temperature = %f' % 1.0)

                # pdb.set_trace()
                training_logs = []
                # Training Loop

                for step in range(init_step, max_steps):
                    log = kge_model.train_step(kge_model, optimizer, train_iterator, args, self)
                    training_logs.append(log)

                    if step >= warm_up_steps:
                        current_learning_rate = current_learning_rate / 10
                        logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                        optimizer = torch.optim.Adam(
                            filter(lambda p: p.requires_grad, kge_model.parameters()),
                            lr=current_learning_rate
                        )
                        warm_up_steps = warm_up_steps * 3

                    if step % args.save_checkpoint_steps == 0:
                        save_variable_list = {
                            'step': step,
                            'current_learning_rate': current_learning_rate,
                            'warm_up_steps': warm_up_steps
                        }

                    if step % args.log_steps == 0:
                        metrics = {}
                        for metric in training_logs[0].keys():
                            metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                        self.log_metrics('Training average', step, metrics)
                        training_logs = []
        if args.do_test:
            logging.info('Evaluating on Test Dataset...')
            metrics, preds = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
            self.log_metrics('Test', step, metrics)
            if args.record:
                # save the final results
                with open(args.save_path + 'result_kge.txt', 'w') as fo:
                    for metric in metrics:
                        fo.write('{} : {}\n'.format(metric, metrics[metric]))

                    # Save the predictions on test data
                with open(args.save_path + 'pred_kge.txt', 'w') as fo:
                    for h, r, t, f, rk, l in preds:
                        fo.write('{}\t{}\t{}\t{}\t{}\n'.format(self.id2entity[h], self.id2relation[r], self.id2entity[t], f, rk))
                        # for e, val in l:
                        #    fo.write('{}:{:.4f} '.format(self.id2entity[e], val))
                        # fo.write('\n')
        if args.record:
            # Annotate hidden triplets
            scores = kge_model.infer_step(kge_model, hidden_triples, args)
            with open(args.save_path + 'annotation.txt', 'w') as fo:
                for (h, r, t), s in zip(hidden_triples, scores):
                    fo.write('{}\t{}\t{}\t{}\n'.format(self.id2entity[h], self.id2relation[r], self.id2entity[t], s))

        if args.evaluate_train:
            logging.info('Evaluating on Training Dataset...')
            metrics, preds = kge_model.test_step(kge_model, train_triplets, all_true_triples, args)
            self.log_metrics('Test', step, metrics)

        file_path = args.save_path + 'annotation.txt'
        pdb.set_trace()
        self.read_probability_of_hidden_triplets(file_path, 0.7)
        self.output_prediction(args.save_path + 'pred_kge.txt')

































