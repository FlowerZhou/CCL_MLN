"""
EM Framework for MLN, according to paper "ExpressGNN"
Instead maximizing the log-likelihood of log(P), optimize the variational evidence lower bound(ELBO)
In the E-Step: infer the posterior distribution of the latent variables
In the M-Step: learn the weights of logic formula in MLN
"""
import torch
from model.mean_field_posterior import FactorizedPosterior
from model.gcn import GCN, TrainableEmbedding
from model.mln import ConditionalMLN
from data_process.dataset import Dataset
from common.cmd_args import cmd_args
from tqdm import tqdm
import torch.optim as optim
from model.graph import KnowledgeGraph
from common.predicate import PRED_DICT
from common.utils import EarlyStopMonitor, get_lr, count_parameters
from common.evaluate import gen_eval_query
from itertools import chain
import random
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import cross_val_score
from os.path import join as joinpath
import os
import math
from collections import Counter
import pdb


class EMFramework(object):
    """
    EM-Framework
    """
    def __init__(self, mln, db):
        self.mln = mln
        self.db = db
        self.db.const_sort_dict = dict([(type_name, sorted(list(self.db.domains[type_name])))
                                   for type_name in self.db.domains.keys()])
        self.kg = KnowledgeGraph(self.db.fact_dict, self.mln._predicates, self.db)
        # GCN(kg, embedding_size-gcn_free_size, gcn_free_size, num_hops, num_layers, transductive)
        self.gcn = GCN(self.kg, 64-32, 32, num_hops=3, num_layers=2, transductive=0).to('cpu')
        # FactorizedPosterior(kg, embedding_size, slice_dim)
        self.posterior_model = FactorizedPosterior(self.kg, 64, 8, self.mln).to('cpu')
        # EarlyStopMonitor(patience)
        self.monitor = EarlyStopMonitor(10)
        self.all_params = chain.from_iterable([self.posterior_model.parameters(), self.gcn.parameters()])
        self.optimizer = optim.Adam(self.all_params, lr=0.0005, weight_decay=0.0)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', factor=0.5, patience=10,
                                                         min_lr=0.00001)
        self.cnt_gcn_params = count_parameters(self.gcn)
        self.cnt_posterior_params = count_parameters(self.posterior_model)
        self.db.data_process()
        self.batch_size = 8
        self.log_path = joinpath(os.getcwd() + '/result/eval.result')

    def em_procedure(self):
        """
        EM Steps, M-Learning, E-Inference
        """
        # pdb.set_trace()
        # prepare data for M-Step Learning step
        tqdm.write("preparing data for M-Step...")
        pred_arg1_set_arg2 = dict()
        pred_arg2_set_arg1 = dict()
        pred_fact_set = dict()
        for pred in self.db.fact_dict_2:
            pred_arg1_set_arg2[pred] = dict()
            pred_arg2_set_arg1[pred] = dict()
            pred_fact_set[pred] = set()
            for _, args in self.db.fact_dict_2[pred]:
                if len(args) == 2:
                    if args[0] not in pred_arg1_set_arg2[pred]:
                        pred_arg1_set_arg2[pred][args[0]] = set()
                    if args[1] not in pred_arg2_set_arg1[pred]:
                        pred_arg2_set_arg1[pred][args[1]] = set()
                    pred_arg1_set_arg2[pred][args[0]].add(args[1])
                    pred_arg2_set_arg1[pred][args[1]].add(args[0])
                    pred_fact_set[pred].add(args)
        # pdb.set_trace()
        grounded_rules = []
        for rule_index, rule in enumerate(self.mln.formulas):
            grounded_rules.append(set())
            body_atoms = []
            head_atoms = None
            if hasattr(rule, 'children'):
                for atom in rule.children:
                    if atom.negated:
                        body_atoms.append(atom)
                    elif head_atoms is None:
                        head_atoms = atom
            else:
                if rule.negated:
                    body_atoms.append(rule)
                elif head_atoms is None:
                    head_atoms = rule

            # atom in body must be observed
            # pdb.set_trace()
            # assert len(body_atoms) <= 2
            if len(body_atoms) > 2:
                continue
            if len(body_atoms) > 0:
                body1 = body_atoms[0]
                for _, body1_args in self.db.fact_dict_2[body1.pred_name]:
                    if len(body1_args) == 2:
                        var2arg = dict()
                        var2arg[body1.args[0]] = body1_args[0]
                        var2arg[body1.args[1]] = body1_args[1]
                        for body2 in body_atoms[1:]:
                            if len(body2.args) == 2:
                                if body2.args[0] in var2arg:
                                    if var2arg[body2.args[0]] in pred_arg1_set_arg2[body2.pred_name]:
                                        for body2_arg2 in pred_arg1_set_arg2[body2.pred_name][var2arg[body2.args[0]]]:
                                            var2arg[body2.args[1]] = body2_arg2
                                            grounded_rules[rule_index].add(tuple(sorted(var2arg.items())))
                                elif body2.args[1] in var2arg:
                                    if var2arg[body2.args[1]] in pred_arg2_set_arg1[body2.pred_name]:
                                        for body2_arg1 in pred_arg2_set_arg1[body2.pred_name][var2arg[body2.args[1]]]:
                                            var2arg[body2.args[0]] = body2_arg1
                                            grounded_rules[rule_index].add(tuple(sorted(var2arg.items())))
        # pdb.set_trace()
        # collect head atoms derived by grounded formulas
        grounded_obs = dict()
        grounded_hid = dict()
        grounded_hid_score = dict()
        cnt_hid = 0
        # pdb.set_trace()
        for rule_index in range(len(self.mln.formulas)):
            rule = self.mln.formulas[rule_index]
            for var2arg in grounded_rules[rule_index]:
                var2arg = dict(var2arg)
                head_atoms = rule.children[-1]
                # assert not head_atoms.negated
                if head_atoms.negated:
                    continue
                # pdb.set_trace()
                pred = head_atoms.pred_name
                if len(head_atoms.args) == 2:
                    args = (var2arg[head_atoms.args[0]], var2arg[head_atoms.args[1]])
                    if args in pred_fact_set[pred]:
                        if (pred, args) not in grounded_obs:
                            grounded_obs[(pred, args)] = []
                        grounded_obs[(pred, args)].append(rule_index)
                        # grounded_obs[(pred, args)] = list(set(grounded_obs[(pred, args)]))
                    else:
                        if (pred, args) not in grounded_hid:
                            grounded_hid[(pred, args)] = []
                        grounded_hid[(pred, args)].append(rule_index)
                        # grounded_hid[(pred, args)] = list(set(grounded_hid[(pred, args)]))
        tqdm.write('observed: %d, hidden: %d' % (len(grounded_obs), len(grounded_hid)))

        # Aggregate atoms by predicates for fast inference
        # pdb.set_trace()
        pred_aggregated_hid = dict()
        pred_aggregated_hid_args = dict()
        for (pred, args) in grounded_hid:
            if pred not in pred_aggregated_hid:
                pred_aggregated_hid[pred] = []
            if pred not in pred_aggregated_hid_args:
                pred_aggregated_hid_args[pred] = []
            pred_aggregated_hid[pred].append((self.db.const2ind[args[0]], self.db.const2ind[args[1]]))
            pred_aggregated_hid_args[pred].append(args)
        pred_aggregated_hid_list = [[pred, pred_aggregated_hid[pred]] for pred in sorted(pred_aggregated_hid.keys())]

        pred_aggregated_hid_args_list = []
        for k in pred_aggregated_hid_args:
            for v in pred_aggregated_hid_args[k]:
                pred_aggregated_hid_args_list.append((k, v))

        for current_epoch in range(100):
            """
            # This part is for traditional MLN dataset, such like UW-CSE
            pbar = tqdm(range(10))
            acc_loss = 0.0
            # pdb.set_trace()
            # E-Step: optimize the parameters in posterior model
            for k in pbar:
                node_embeds = self.gcn(self.db)
                batch_neg_mask, flat_list, batch_latent_var_inds, observed_rule_cnts, batch_observed_vars = \
                    self.db.get_batch_rnd(
                        observed_prob=0.9,
                        filter_latent=0,
                        closed_world=0,
                        filter_observed=1
                    )
                posterior_prob = self.posterior_model(flat_list, node_embeds)
                entropy = self.compute_entropy(posterior_prob) / 1
                entropy = entropy.to('cpu')
                posterior_prob = posterior_prob.to('cpu')
                # pdb.set_trace()
                potential = self.mln.forward(batch_neg_mask, batch_latent_var_inds, observed_rule_cnts, posterior_prob,
                                             flat_list, batch_observed_vars)
                self.optimizer.zero_grad()
                loss = - (potential + entropy) / 16
                acc_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                pbar.set_description('train loss: %.4f, lr: %.4g' % (acc_loss / (k + 1), get_lr(self.optimizer)))
            """
            # This part is for dataset like cora
            # pdb.set_trace()
            num_batches = int(math.ceil(len(self.db.test_fact_ls)/self.batch_size))
            pbar = tqdm(total=num_batches)
            acc_loss = 0.0
            cur_batch = 0
            for samples_by_r, latent_mask_by_r, neg_mask_by_r, obs_var_by_r, neg_var_by_r in \
                    self.db.get_batch_by_q(self.batch_size):

                node_embeds = self.gcn(self.db)
                loss = 0.0
                r_cnt = 0
                for ind, samples in enumerate(samples_by_r):
                    neg_mask = neg_mask_by_r[ind]
                    latent_mask = latent_mask_by_r[ind]
                    obs_var = obs_var_by_r[ind]
                    neg_var = neg_var_by_r[ind]

                    if sum([len(e[1]) for e in neg_mask]) == 0:
                        continue
                    potential, posterior_prob, obs_xent = self.posterior_model([samples, neg_mask, latent_mask, obs_var,
                                                                                neg_var], node_embeds, fast_mode=True)
                    entropy = self.compute_entropy(posterior_prob) / 1
                    loss += - (potential.sum() * self.mln.formulas[ind].weight + entropy) / \
                            (potential.size(0) + 1e-6) + obs_xent
                    r_cnt += 1
                if r_cnt > 0:
                    loss /= r_cnt
                    acc_loss += loss.item()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                pbar.update()
                cur_batch += 1
                pbar.set_description(
                    'Epoch %d, train loss: %.4f, lr: %.4g' % (current_epoch, acc_loss / cur_batch, get_lr(self.optimizer)))
                # pdb.set_trace()


            # pdb.set_trace()
            # pred_aggregated_hid_args_list = [(k, v) for k, v in pred_aggregated_hid_args.items()]
            # M-Step: optimize the weights of logic rules
            with torch.no_grad():
                posterior_prob = self.posterior_model(pred_aggregated_hid_list, node_embeds, fast_inference_mode=True)
                for pred_i, (pred, var_ls) in enumerate(pred_aggregated_hid_list):
                    for var_i, var in enumerate(var_ls):
                        args = pred_aggregated_hid_args[pred][var_i]
                        grounded_hid_score[(pred, args)] = posterior_prob[pred_i][var_i]
                        # grounded_hid_score[(pred, args)] = posterior_prob[pred_i]

                rule_weight_gradient = torch.zeros(len(self.mln.formulas))
                for (pred, args) in grounded_obs:
                    # pdb.set_trace()
                    for rule_idx in set(grounded_obs[(pred, args)]):
                        rule_weight_gradient[rule_idx] += 1.0 - self.compute_mb_proba(self.mln.formulas,
                                                                                      grounded_obs[(pred, args)])
                for (pred, args) in grounded_hid:
                    for rule_idx in set(grounded_hid[(pred, args)]):
                        target = grounded_hid_score[(pred, args)]
                        rule_weight_gradient[rule_idx] += target - self.compute_mb_proba(self.mln.formulas,
                                                                                         grounded_hid[(pred, args)])

                # pdb.set_trace()
                for rule_idx, rule in enumerate(self.mln.formulas):
                    rule.weight += cmd_args.learning_rate_rule_weights * rule_weight_gradient[rule_idx]
                    print(self.mln.formulas[rule_idx].weight, end=' ')
            pbar.close()

            # generate rank list
            node_embeds = self.gcn(self.db)
            pbar = tqdm(total=len(self.db.test_fact_ls))
            pbar.write('*' * 10 + ' Evaluation ' + '*' * 10)
            rrank = 0.0
            hits = 0.0
            cnt = 0

            rrank_pred = dict([(pred_name, 0.0) for pred_name in self.mln.predicate_name])
            hits_pred = dict([(pred_name, 0.0) for pred_name in self.mln.predicate_name])
            cnt_pred = dict([(pred_name, 0.0) for pred_name in self.mln.predicate_name])
            # pdb.set_trace()
            for pred_name, X, invX, sample in gen_eval_query(self.db, const2ind=self.kg.ent2idx):
                x_mat = np.array(X)
                invx_mat = np.array(invX)
                sample_mat = np.array(sample)

                tail_score, head_score, true_score = self.posterior_model([pred_name, x_mat, invx_mat, sample_mat],
                                                                     node_embeds,
                                                                     batch_mode=True)

                rank = torch.sum(tail_score >= true_score).item() + 1
                rrank += 1.0 / rank
                hits += 1 if rank <= 10 else 0

                rrank_pred[pred_name] += 1.0 / rank
                hits_pred[pred_name] += 1 if rank <= 10 else 0

                rank = torch.sum(head_score >= true_score).item() + 1
                rrank += 1.0 / rank
                hits += 1 if rank <= 10 else 0

                rrank_pred[pred_name] += 1.0 / rank
                hits_pred[pred_name] += 1 if rank <= 10 else 0

                cnt_pred[pred_name] += 2
                cnt += 2

                if cnt % 100 == 0:
                    with open(self.log_path, 'w') as f:
                        f.write('%i sample eval\n' % cnt)
                        f.write('mmr %.4f\n' % (rrank / cnt))
                        f.write('hits %.4f\n' % (hits / cnt))

                        f.write('\n')
                        for pred_name in PRED_DICT:
                            if cnt_pred[pred_name] == 0:
                                continue
                            f.write('mmr %s %.4f\n' % (pred_name, rrank_pred[pred_name] / cnt_pred[pred_name]))
                            f.write('hits %s %.4f\n' % (pred_name, hits_pred[pred_name] / cnt_pred[pred_name]))

                pbar.update()

            with open(self.log_path, 'w') as f:
                f.write('complete\n')
                f.write('mmr %.4f\n' % (rrank / cnt))
                f.write('hits %.4f\n' % (hits / cnt))
                f.write('\n')

                tqdm.write('mmr %.4f\n' % (rrank / cnt))
                tqdm.write('hits %.4f\n' % (hits / cnt))

                for pred_name in PRED_DICT:
                    if cnt_pred[pred_name] == 0:
                        continue
                    f.write('mmr %s %.4f\n' % (pred_name, rrank_pred[pred_name] / cnt_pred[pred_name]))
                    f.write('hits %s %.4f\n' % (pred_name, hits_pred[pred_name] / cnt_pred[pred_name]))

            os.system('mv %s %s' % (self.log_path, joinpath(cmd_args.exp_path,
                                                      'performance_hits_%.4f_mmr_%.4f.txt' % (
                                                      (hits / cnt), (rrank / cnt)))))
            pbar.close()
        pdb.set_trace()

        """
        # this part is for traditional MLN inference
        # test do inference
        node_embeds = self.gcn(self.db)
        # pdb.set_trace()
        with torch.no_grad():
            posterior_prob = self.posterior_model([(e[1], e[2]) for e in self.db.test_fact_ls], node_embeds)
            posterior_prob = posterior_prob.to('cpu')
            print("***************************************************************")
            print(posterior_prob)
            print("***************************************************************")
            # label must have at least 2 dimensions and at least have 2 different types or will report an error
            label = np.array([e[0] for e in self.db.test_fact_ls])
            test_log_prob = float(np.sum(np.log(np.clip(np.abs((1 - label) - posterior_prob.numpy()),
                                                            1e-6, 1 - 1e-6))))

            my_metric = 'roc_auc'

            auc_roc = roc_auc_score(label, posterior_prob.numpy())
            auc_pr = average_precision_score(label, posterior_prob.numpy())
            tqdm.write('Epoch: %d, train loss: %.4f, test auc-roc: %.4f, test auc-pr: %.4f, test log prob: %.4f' %
                           (current_epoch, acc_loss / cmd_args.num_batches, auc_roc, auc_pr, test_log_prob))
        # validation for early stop
        # pdb.set_trace()
        valid_sample = []
        valid_label = []
        for pred_name in self.db.valid_dict_2:
            for val, consts in self.db.valid_dict_2[pred_name]:
                valid_sample.append((pred_name, consts))
                valid_label.append(val)
        valid_label = np.array(valid_label)
        valid_prob = self.posterior_model(valid_sample, node_embeds)
        valid_prob = valid_prob.to('cpu')
        self.evaluation()
        """

    def evaluation(self):
        """
        evaluation after training
        """
        pdb.set_trace()
        node_embeds = self.gcn(self.db)
        with torch.no_grad():
            posterior_prob = self.posterior_model([(e[1], e[2]) for e in self.db.test_fact_ls], node_embeds)
            posterior_prob = posterior_prob.to('cpu')
            label = np.array([e[0] for e in self.db.test_fact_ls])
            test_log_prob = float(np.sum(np.log(np.clip(np.abs((1 - label) - posterior_prob.numpy()), 1e-6, 1 - 1e-6))))
            auc_roc = roc_auc_score(label, posterior_prob.numpy())
            auc_pr = average_precision_score(label, posterior_prob.numpy())
            tqdm.write('test auc-roc: %.4f, test auc-pr: %.4f, test log prob: %.4f' % (auc_roc, auc_pr, test_log_prob))

    def compute_entropy(self, posterior_prob):
        eps = 1e-6
        posterior_prob.clamp_(eps, 1-eps)
        compl_prob = 1 - posterior_prob
        entropy = -(posterior_prob * torch.log(posterior_prob) + compl_prob * torch.log(compl_prob)).sum()
        return entropy

    def compute_mb_proba(self, rule_ls, ls_rule_index):
        rule_index_cnt = Counter(ls_rule_index)
        numerator = 0
        for rule_idx in rule_index_cnt:
            weight = rule_ls[rule_idx].weight
            cnt = rule_index_cnt[rule_idx]
            numerator += math.exp(weight * cnt)
        return numerator / (numerator + 1.0)






















