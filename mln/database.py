import logging
import sys
import re
import traceback
from logic.elements import Logic
from io import StringIO
from dnutils import logs, ifnone, out
from dnutils.console import barstr
from mln.util import merge_dom
from collections import defaultdict
from mln.project import MLNPath
from mln.util import strip_comments
from logic.fol import FirstOrderLogic
from collections import Counter
from random import shuffle, choice
import os
import numpy as np
import itertools
import torch.nn as nn
import torch
import random
import pdb
logger = logging.getLogger(__name__)

# grounded rule stats code
BAD = 0  # sample not valid
FULL_OBSERVERED = 1  # sample valid, but rule contains only observed vars and does not have negation for all atoms
GOOD = 2  # sample valid


class DataBase(object):
    """
    Represents an MLN Database, which consists of a set of ground atoms, each assigned a truth value.


    :member mln:            the respective :class:`mln.base.MLN` object that this Database is associated with
    :member domains:        dict mapping the variable domains specific to this data base (i.e. without
                            values from the MLN domains which are not present in the DB) to the set possible values.
    :member evidence:       dictionary mapping ground atom strings to truth values.
    :param mln:             the :class:`mln.base.MLN` instance that the database shall be associated with.
    :param evidence:        a dictionary mapping ground atoms to their truth values.
    :param db_file:          if specified, a database is loaded from the given file path.
    :param ignore_unknown_preds: see :func:`mln.database.parse_db`
    """

    def __init__(self, mln, evidence=None, db_file=None, ignore_unknown_preds=False):
        self.mln = mln
        self._domains = defaultdict(list)
        self._evidence = {}
        if db_file is not None:
            DataBase.load(mln, db_file, db=self, ignore_unknown_preds=ignore_unknown_preds)
        if evidence is not None:
            for atom, truth in evidence.items():
                self.add(atom, truth)
        # add according to ExpressGNN
        # pdb.set_trace()
        self.fact_dict = dict((pred_name, set()) for pred_name in self.mln.predicate_name)
        self.test_fact_dict = dict((pred_name, set()) for pred_name in self.mln.predicate_name)
        self.valid_dict = dict((pred_name, set()) for pred_name in self.mln.predicate_name)
        self.ht_dict = dict((pred_name, [dict(), dict()]) for pred_name in self.mln.predicate_name)
        self.ht_dict_train = dict((pred_name, [dict(), dict()]) for pred_name in self.mln.predicate_name)
        self.fact_dict_2 = dict()
        self.valid_dict_2 = dict()
        self.atom_key_dict_ls = []  # predicate_name, args, atom-key dict
        self.test_fact_ls = []
        self.valid_fact_ls = []
        const_cnter = Counter
        self.const_sort_dict = dict()
        self.const2ind = dict()
        self.batch_size = 16
        self.shuffle_sampling = 1
        self.mln.rule_weights_lin = nn.Linear(len(self.mln.formulas), 1, bias=False)
        # pdb.set_trace()
        self.mln.rule_weights_lin.weight = nn.Parameter(torch.tensor([[int(float(weight)) for weight in self.mln.weights]],
                                                                     dtype=torch.float))

    def data_process(self):
        # pdb.set_trace()
        self.const_sort_dict = dict([(type_name, sorted(list(self.domains[type_name]))) for
                                     type_name in self.mln.domains.keys()])
        # pdb.set_trace()
        i = 0
        for key in self.const_sort_dict.keys():
            for const in self.const_sort_dict[key]:
                self.const2ind[const] = i
                i += 1

        # self.const2ind = dict([(const, i) for i, const in enumerate(self.const_sort_dict['type'])])
        self.add_atom_key()

    def add_atom_key(self):
        # pdb.set_trace()
        for f in self.mln.formulas:
            atom_key_dict = dict()
            if hasattr(f, 'children'):
                for atom in f.children:
                    atom_dict = dict((var_name, dict()) for var_name in atom.args)
                    for i, var_name in enumerate(atom.args):
                        if atom.pred_name not in self.fact_dict:
                            continue
                        for v in self.fact_dict[atom.pred_name]:
                            if v[1][i] not in atom_dict[var_name]:
                                atom_dict[var_name][v[1][i]] = [v]
                            else:
                                atom_dict[var_name][v[1][i]] += [v]
                    # happens if predicate occurs more than once in one rule then we merge the set
                    if atom.pred_name in atom_key_dict:
                        for k, v in atom_dict.items():
                            if k not in atom_key_dict[atom.pred_name]:
                                atom_key_dict[atom.pred_name][k] = v
                    else:
                        atom_key_dict[atom.pred_name] = atom_dict
            else:
                atom = f
                atom_dict = dict((var_name, dict()) for var_name in atom.args)
                for i, var_name in enumerate(atom.args):
                    if atom.pred_name not in self.fact_dict:
                        continue
                    for v in self.fact_dict[atom.pred_name]:
                        if v[1][i] not in atom_dict[var_name]:
                            atom_dict[var_name][v[1][i]] = [v]
                        else:
                            atom_dict[var_name][v[1][i]] += [v]
                # happens if predicate occurs more than once in one rule then we merge the set
                if atom.pred_name in atom_key_dict:
                    for k, v in atom_dict.items():
                        if k not in atom_key_dict[atom.pred_name]:
                            atom_key_dict[atom.pred_name][k] = v
                else:
                    atom_key_dict[atom.pred_name] = atom_dict
            self.atom_key_dict_ls.append(atom_key_dict)

    def add_ht(self, pn, c_ls, ht_dict, flag):
        if flag == 0:
            if c_ls[0] in ht_dict[pn][0]:
                ht_dict[pn][0][c_ls[0]].add(c_ls[0])
            else:
                ht_dict[pn][0][c_ls[0]] = set([c_ls[0]])
        elif flag == 1:
            if c_ls[0] in ht_dict[pn][0]:
                ht_dict[pn][0][c_ls[0]].add(c_ls[1])
            else:
                ht_dict[pn][0][c_ls[0]] = set([c_ls[1]])

            if c_ls[1] in ht_dict[pn][1]:
                ht_dict[pn][1][c_ls[1]].add(c_ls[0])
            else:
                ht_dict[pn][1][c_ls[1]] = set([c_ls[0]])

    def get_batch_rnd(self, observed_prob=0.7, filter_latent=True, closed_world=False, filter_observed=False):
        """
        return a batch of ground formula by random sampling with controllable bias towards those containing observed
        variables. The overall sampling logic is that:
        (1) rnd sample a rule from rule_ls
        (2) shuffle the predicates contained in the rule
        (3) for each of these predicates, with (observed_prob) it will be instantiated as observed variables, and for
        (1-observed_prob) if will be simply uniformly instantiated.
            > if observed var, the sample from the KB, which is self.fact_dict, if failed for any reason, go to the next
            > if uniformly sample, then for each logic variable in the predicate, instantiate it with a uniform sample
            from the corresponding constant dict
        :param observed_prob: probability of instantiating a predicate as observed variable
        :param filter_latent: filter out ground formula containing only latent variables
        :param closed_world: if set true, reduce the sampling space of all predicates not in the test_dict to the set
            specified in fact_dict
        :param filter_observed: filter out ground formula containing only observed variables
        """
        # pdb.set_trace()
        batch_neg_mask = [[] for _ in range(len(self.mln.formulas))]
        batch_latent_var_inds = [[] for _ in range(len(self.mln.formulas))]
        batch_observed_vars = [[] for _ in range(len(self.mln.formulas))]
        observed_rule_cnts = [0.0 for _ in range(len(self.mln.formulas))]
        flat_latent_vars = dict()
        cnt = 0
        inds = list(range(len(self.mln.formulas)))
        while cnt < self.batch_size:
            # randomly sample a formula
            if self.shuffle_sampling:
                shuffle(inds)
            for index in inds:
                rule = self.mln.formulas[index]
                atom_key_dict = self.atom_key_dict_ls[index]
                sub = [None] * len(rule.var_doms())
                # randomly sample an atom from the formula
                if hasattr(rule, 'children'):
                    atom_inds = list(range(len(rule.children)))
                else:
                    atom_inds = list(range(1))
                shuffle(atom_inds)
                for atom_index in atom_inds:
                    if hasattr(rule, 'children'):
                        atom = rule.children[atom_index]
                    else:
                        atom = rule
                    atom_dict = atom_key_dict[atom.pred_name]

                    # instantiate the predicate
                    self._instantiate_predicate(atom, atom_dict, sub, rule, observed_prob)
                    # if variable substitution is complete already then exit
                    if not (None in sub):
                        break
                # generate latent and observed var labels and their negation masks
                latent_vars, observed_vars, latent_neg_mask, \
                observed_neg_mask = self._gen_mask(rule, sub, closed_world)
                # check sampled ground rule status
                stat_code = self._get_rule_state(observed_vars, latent_vars, observed_neg_mask,
                                                 filter_latent, filter_observed)

                # is a valid sample with only observed vars and does not have negation on all of them
                if stat_code == FULL_OBSERVERED:
                    observed_rule_cnts[index] += 1
                    cnt += 1
                # is a valid sample
                elif stat_code == GOOD:
                    batch_neg_mask[index].append([latent_neg_mask, observed_neg_mask])
                    for latent_var in latent_vars:
                        if latent_var not in flat_latent_vars:
                            flat_latent_vars[latent_var] = len(flat_latent_vars)
                    batch_latent_var_inds[index].append([flat_latent_vars[e] for e in latent_vars])
                    batch_observed_vars[index].append(observed_vars)
                    cnt += 1
                # not a valid sample
                else:
                    continue
                if cnt >= self.batch_size or cnt > len(self.mln._formulas):
                    break
        flat_list = sorted([(k, v) for k, v in flat_latent_vars.items()], key=lambda x: x[1])
        flat_list = [e[0] for e in flat_list]
        return batch_neg_mask, flat_list, batch_latent_var_inds, observed_rule_cnts, batch_observed_vars

    def get_batch_by_q(self, batchsize, observed_prob=1.0, validation=False):

        # pdb.set_trace()
        samples_by_r = [[[atom.pred_name, []] for atom in rule.children] for _, rule in self.mln.iter_formulas()]
        neg_mask_by_r = [[[atom.pred_name, []] for atom in rule.children] for _, rule in self.mln.iter_formulas()]
        latent_mask_by_r = [[[atom.pred_name, []] for atom in rule.children] for _, rule in self.mln.iter_formulas()]
        obs_var_by_r = [[[atom.pred_name, []] for atom in rule.children] for _, rule in self.mln.iter_formulas()]
        neg_var_by_r = [[[atom.pred_name, []] for atom in rule.children] for _, rule in self.mln.iter_formulas()]
        cnt = 0
        num_ents = len(self.const2ind)
        ind2const = []
        for k in self.const2ind.keys():
            ind2const.append(k)

        def gen_fake(c1, c2, pn):
            # pdb.set_trace()
            for _ in range(10):
                c1_fake = random.randint(0, num_ents - 1)
                c2_fake = random.randint(0, num_ents - 1)
                if np.random.rand() > 0.5:
                    if ind2const[c1_fake] not in self.ht_dict_train[pn][1].get(ind2const[c2]):
                        return c1_fake, c2
                    else:
                        if ind2const[c2_fake] not in self.ht_dict_train[pn][0].get(ind2const[c1]):
                            return c1, c2_fake
            return None, None

        if validation:
            fact_ls = self.valid_fact_ls
        else:
            fact_ls = self.test_fact_ls

        # pdb.set_trace()
        for val, pred_name, consts in fact_ls:
            for rule_i, rule in self.mln.iter_formulas():
                # find rule with pred_name as head
                if rule.children[-1].pred_name != pred_name:
                    continue
                samples = samples_by_r[rule_i]
                neg_mask = neg_mask_by_r[rule_i]
                latent_mask = latent_mask_by_r[rule_i]
                obs_var = obs_var_by_r[rule_i]
                neg_var = neg_var_by_r[rule_i]

                key2ind = dict(zip(rule.var_doms().keys(), range(len(rule.var_doms().keys()))))
                var2ind = key2ind
                var2type = rule.var_doms()
                # pdb.set_trace()
                sub = [None] * len(rule.var_doms())   # substitutions
                if len(rule.children[-1].args) > 1:
                    vn0, vn1 = rule.children[-1].args
                    sub[var2ind[vn0]] = consts[0]
                    sub[var2ind[vn1]] = consts[1]
                else:
                    vn0 = rule.children[-1].args[0]
                    sub[var2ind[vn0]] = consts[0]

                sample_buff = [[] for _ in rule.children]
                neg_mask_buff = [[] for _ in rule.children]
                latent_mask_buff = [[] for _ in rule.children]

                atom_inds = list(range(len(rule.children) - 1))
                shuffle(atom_inds)
                succ = True
                obs_list = []
                for atom_index in atom_inds:
                    atom = rule.children[atom_index]
                    pred_ht_dict = self.ht_dict_train[atom.pred_name]
                    gen_latent = np.random.rand() > observed_prob
                    c_ls, islatent, atom_succ = self._inst_var(sub, var2ind, var2type, atom, pred_ht_dict, gen_latent)

                    assert atom_succ
                    if not islatent:
                        obs_var[atom_index][1].append(c_ls)
                        c1, c2 = gen_fake(c_ls[0], c_ls[1], atom.pred_name)
                        if c1 is not None:
                            neg_var[atom_index][1].append([c1, c2])
                    succ = succ and atom_succ
                    obs_list.append(not islatent)
                    sample_buff[atom_index].append(c_ls)
                    latent_mask_buff[atom_index].append(1 if islatent else 0)
                    neg_mask_buff[atom_index].append(0 if atom.negated else 1)

                if succ and any(obs_list):
                    for i in range(len(rule.children)):
                        samples[i][1].extend(sample_buff[i])
                        latent_mask[i][1].extend(latent_mask_buff[i])
                        neg_mask[i][1].extend(neg_mask_buff[i])
                    samples[-1][1].append([self.const2ind[consts[0]], self.const2ind[consts[1]]])
                    latent_mask[-1][1].append(1)
                    neg_mask[-1][1].append(1)
                    cnt += 1
            if cnt >= batchsize:
                yield samples_by_r, latent_mask_by_r, neg_mask_by_r, obs_var_by_r, neg_var_by_r
                samples_by_r = [[[atom.pred_name, []] for atom in rule.children] for _, rule in self.mln.iter_formulas()]
                neg_mask_by_r = [[[atom.pred_name, []] for atom in rule.children] for _, rule in self.mln.iter_formulas()]
                latent_mask_by_r = [[[atom.pred_name, []] for atom in rule.children] for _, rule in self.mln.iter_formulas()]
                obs_var_by_r = [[[atom.pred_name, []] for atom in rule.children] for _, rule in self.mln.iter_formulas()]
                neg_var_by_r = [[[atom.pred_name, []] for atom in rule.children] for _, rule in self.mln.iter_formulas()]
                cnt = 0
        # pdb.set_trace()
        yield samples_by_r, latent_mask_by_r, neg_mask_by_r, obs_var_by_r, neg_var_by_r

    # TODO only binary | only positive fact!
    def _inst_var(self, sub, var2ind, var2type, at, ht_dict, gen_latent):
        # pdb.set_trace()
        if len(at.args) != 2:
            raise KeyError
        must_latent = gen_latent
        if must_latent:
            tmp = [sub[var2ind[vn]] for vn in at.args]
            for i, subi in enumerate(tmp):
                if subi is None:
                    tmp[i] = random.choice(self.const_sort_dict[var2type[at.args[i]]])
            islatent = (tmp[0] not in ht_dict[0]) or (tmp[1] not in ht_dict[0][tmp[0]])
            for i, vn in enumerate(at.args):
                sub[var2ind][vn] = tmp[i]
            return [self.const2ind[subi] for subi in tmp], islatent, islatent or at.negated
        vn0 = at.args[0]
        sub0 = sub[var2ind[vn0]]
        vn1 = at.args[1]
        sub1 = sub[var2ind[vn1]]

        if sub0 is None:
            if sub1 is None:
                if len(ht_dict[0]) > 0:
                    sub0 = random.choice(tuple(ht_dict[0].keys()))
                    sub1 = random.choice(tuple(ht_dict[0][sub0]))
                    sub[var2ind[vn0]] = sub0
                    sub[var2ind[vn1]] = sub1
                    return [self.const2ind[sub0], self.const2ind[sub1]], False, at.negated
            else:
                if sub1 in ht_dict[1]:
                    sub0 = random.choice(tuple(ht_dict[1][sub1]))
                    sub[var2ind[vn0]] = sub0
                    return [self.const2ind[sub0], self.const2ind[sub1]], False, at.negated
                else:
                    sub0 = random.choice(self.const_sort_dict[var2type[vn0]])
                    sub[var2ind[vn0]] = sub0
                    return [self.const2ind[sub0], self.const2ind[sub1]], True, True
        else:
            if sub1 is None:
                if sub0 in ht_dict[0]:
                    sub1 = random.choice(tuple(ht_dict[0][sub0]))
                    sub[var2ind[vn1]] = sub1
                    return [self.const2ind[sub0], self.const2ind[sub1]], False, at.negated
                else:
                    sub1 = random.choice(self.const_sort_dict[var2type[vn1]])
                    sub[var2ind[vn1]] = sub1
                    return [self.const2ind[sub0], self.const2ind[sub1]], True, True
            else:
                islatent = (sub0 not in ht_dict[0]) or (sub1 not in ht_dict[0][sub0])
                return [self.const2ind[sub0], self.const2ind[sub1]], islatent, islatent or at.negated

    def _get_rule_state(self, observed_vars, latent_vars, observed_neg_mask, filter_latent, filter_observed):
        """

        """
        is_full_latent = len(observed_vars) == 0
        is_full_observed = len(latent_vars) == 0
        if is_full_latent and filter_latent:
            return BAD
        if is_full_observed:
            if filter_observed:
                return BAD
            is_full_neg = sum(observed_neg_mask) == 0
            if is_full_neg:
                return BAD
            else:
                return FULL_OBSERVERED
        # If observed var already yields 1
        if sum(observed_neg_mask) > 0:
            return BAD
        return GOOD

    def _gen_mask(self, rule, sub, closed_world):
        """
        generate mask
        """
        latent_vars = []
        observed_vars = []  # (true/false, predicate_name)
        latent_neg_mask = []
        observed_neg_mask = []  # record if the atom is negated
        key2ind = dict(zip(rule.var_doms().keys(), range(len(rule.var_doms().keys()))))
        if hasattr(rule, 'children'):
            for atom in rule.children:
                try:
                    grounding = tuple(sub[key2ind[var_name]] for var_name in atom.args)
                except KeyError:
                    continue
                pos_gnding, neg_gnding = (1, grounding), (0, grounding)  # positive and negative
                if pos_gnding in self.fact_dict[atom.pred_name]:
                    observed_vars.append((1, atom.pred_name))
                    observed_neg_mask.append(0 if atom.negated else 1)
                elif neg_gnding in self.fact_dict[atom.pred_name]:
                    observed_vars.append((0, atom.pred_name))
                    observed_neg_mask.append(1 if atom.negated else 0)
                else:
                    if closed_world and (len(self.test_fact_dict[atom.pred_name]) == 0):
                        observed_vars.append((0, atom.pred_name))
                        observed_neg_mask.append(1 if atom.negated else 0)
                    else:
                        latent_vars.append((atom.pred_name, grounding))
                        latent_neg_mask.append(1 if atom.negated else 0)
        else:
            atom = rule
            grounding = tuple(sub[key2ind[var_name]] for var_name in atom.args)
            pos_gnding, neg_gnding = (1, grounding), (0, grounding)  # positive and negative
            if pos_gnding in self.fact_dict[atom.pred_name]:
                observed_vars.append((1, atom.pred_name))
                observed_neg_mask.append(0 if atom.negated else 1)
            elif neg_gnding in self.fact_dict[atom.pred_name]:
                observed_vars.append((0, atom.pred_name))
                observed_neg_mask.append(1 if atom.negated else 0)
            else:
                if closed_world and (len(self.test_fact_dict[atom.pred_name]) == 0):
                    observed_vars.append((0, atom.pred_name))
                    observed_neg_mask.append(1 if atom.negated else 0)
                else:
                    latent_vars.append((atom.pred_name, grounding))
                    latent_neg_mask.append(1 if atom.negated else 0)

        return latent_vars, observed_vars, latent_neg_mask, observed_neg_mask

    def _instantiate_predicate(self, atom, atom_dict, sub, rule, observed_prob):
        """
        instantiate predicate
        """
        key2ind = dict(zip(rule.var_doms().keys(), range(len(rule.var_doms().keys()))))
        rule_vars = rule.var_doms()
        # substitute with observed fact
        if np.random.rand() < observed_prob:
            fact_choice_set = None
            for var_name in atom.args:
                try:
                    const = sub[key2ind[var_name]]
                except KeyError:
                    continue
                if const is None:
                    choice_set = itertools.chain.from_iterable([v for k, v in atom_dict[var_name].items()])
                else:
                    if const in atom_dict[var_name]:
                        choice_set = atom_dict[var_name][const]
                    else:
                        choice_set = []
                if fact_choice_set is None:
                    fact_choice_set = set(choice_set)
                else:
                    fact_choice_set = fact_choice_set.intersection(set(choice_set))
                if len(fact_choice_set) == 0:
                    break
            if len(fact_choice_set) == 0:
                for var_name in atom.args:
                    try:
                        if sub[key2ind[var_name]] is None:
                            sub[key2ind[var_name]] = choice(self.const_sort_dict[rule_vars[var_name]])
                    except KeyError:
                        continue
            else:
                val, const_ls = choice(sorted(list(fact_choice_set)))
                for var_name, const in zip(atom.args, const_ls):
                    try:
                        sub[key2ind[var_name]] = const
                    except KeyError:
                        continue
            # substitute with random facts
        else:
            for var_name in atom.args:
                try:
                    if sub[key2ind[var_name]] is None:
                        sub[key2ind[var_name]] = choice(self.const_sort_dict[rule_vars[var_name]])
                except KeyError:
                    continue

    @property
    def domains(self):
        return self._domains

    @domains.setter
    def domains(self, dom):
        self._domains = dom

    @property
    def evidence(self):
        return self._evidence

    def _atom_str(self, ground_atom):
        """
        converts ground_atom into a valid ground atom in string representation
        """
        if type(ground_atom) is str:
            _, pred_name, args = self.mln.logic.parse_literal(ground_atom)
            atom_str = str(self.mln.logic.ground_atom(pred_name, args, self.mln))
        elif isinstance(ground_atom, Logic.GroundAtom):
            atom_str = str(ground_atom)
        elif isinstance(ground_atom, Logic.GroundLiteral):
            atom_str = str(ground_atom.ground_atom)
            pred_name = ground_atom.ground_atom.pred_name
            args = ground_atom.ground_atom.params
        elif isinstance(ground_atom, Logic.Literal):
            atom_str = str(self.mln.logic.ground_atom(ground_atom.pred_name, ground_atom.args, self.mln))
        return atom_str

    def truth(self, ground_atom):
        """
        Returns the evidence truth value of the given ground atom.

        :param ground_atom:    a ground atom string
        :returns:          the truth value of the ground atom, or None if it is not specified.
        """
        atom_str = self._atom_str(ground_atom)
        return self._evidence.get(atom_str)

    def domain(self, domain):
        """
        Returns the list of domain values of the given domain, or None if no domain
        with the given name exists. If domain is dict, the domain values will be
        updated and the domain will be created, if necessary.

        :param domain:     the name of the domain to be returned.
        """
        if type(domain) is dict:
            for dom_name, values, in domain.items():
                if type(values) is not list:
                    values = [values]
                    dom = self.domain(dom_name)
                    if dom is None:
                        dom = []
                        self._domains[dom_name] = dom
                    for value in values:
                        if value not in dom:
                            dom.append(value)
        elif domain is not None:
            return self._domains.get(domain)
        else:
            return self._domains

    def copy(self, mln=None):
        if mln is None:
            mln = self.mln
        db = DataBase(mln)
        for atom, truth, in self.ground_atom():
            db.add(atom, truth)
        return db

    def union(self, dbs, mln=None):
        """
        Returns a new database consisting of the union of all databases
        given in the arguments. If mln is specified, the new database will
        be attached to that one, otherwise the mln of this database will
        be used.
        """
        db_ = DataBase(mln if mln is not None else self.mln)
        if type(db_) is list:
            dbs = [list(d) for d in dbs] + list(self)
        for atom, truth in dbs:
            db_ << (atom, truth)
        return db_

    def ground_atom(self, pred_names=None):
        """
        Iterates over all ground atoms in this database that match any of
        the given predicate names. If no predicate name is specified, it
        yields all ground atoms in the database.

        :param pred_names:    a list of predicate names that this iteration should be filtered by.
        :returns:            a generator of (atom, truth) tuples.
        """
        for atom, truth in self:
            if pred_names is not None:
                _, pred_name, _ = self.mln.logic.parse_literal(atom)
                if pred_name not in pred_names:
                    continue
            yield atom, truth

    def add(self, ground_literal, truth=1):
        """
        Adds the fact represented by the ground atom, which might be
        a GroundLiteral object or a string.
        """
        if isinstance(ground_literal, str):
            true, pred_name, args = self.mln.logic.parse_literal(ground_literal)
            atom_str = str(self.mln.logic.ground_atom(pred_name, args, self.mln))
        elif isinstance(ground_literal, Logic.GroundLiteral):
            atom_str = str(ground_literal.ground_atom)
            true = not ground_literal.negated
            pred_name = ground_literal.ground_atom.pred_name
            args = ground_literal.ground_atom.args
        if truth in (True, False):
            truth = {True: 1, False: 0}[truth]
        truth = truth if true else 1 - truth
        truth = eval('%.6f' % truth)
        pred = self.mln.predicate(pred_name)
        for dom_name, arg in zip(pred.arg_doms, args):
            self.domain({dom_name: arg})
        # pdb.set_trace()
        self._evidence[atom_str] = truth
        # add according to EM
        self.fact_dict[pred_name].add((truth, tuple(args)))
        self.add_ht(pred_name, args, self.ht_dict, 1)
        self.add_ht(pred_name, args, self.ht_dict_train, 1)
        self.fact_dict_2 = dict((pred_name, sorted(list(self.fact_dict[pred_name])))for pred_name
                                in self.fact_dict.keys())
        self.valid_dict_2 = dict((pred_name, sorted(list(self.valid_dict[pred_name])))for pred_name
                                 in self.valid_dict.keys())
        return self

    def write(self, stream=sys.stdout, color=None, bars=False):
        """
        Writes this database into the stream in the MLN Database format.
        The stream must provide a `write()` method as file objects do.
        """
        if color is None:
            if stream != sys.stdout:
                color = False
            else:
                color = True
        for atom in sorted(self._evidence):
            truth = self._evidence[atom]
            pred, params = self.mln.logic.parse_atom(atom)
            pred = str(pred)
            params = list(map(str, params))
            if bars:
                bar = barstr(30, truth, color='magenta' if color else None)
            else:
                bar = ''
            strout = '%s  %s\n' % (bar if bars else '%.6f' % truth, FirstOrderLogic.Literal(False, pred, params, self.mln))
            stream.write(strout)

    def is_hard(self):
        """
        Determines whether or not this database contains exclusively
        hard evidences.
        """
        return any(map(lambda x: x != 1 and x != 0, self._evidence))

    def retract(self, ground_atom):
        """
        Removes the evidence of the given ground atom in this database.
        """
        if type(ground_atom) is str:
            _, pred_name, args = self.mln.logic.parse_literal(ground_atom)
            atom_str = str(self.mln.logic.ground_atom(pred_name, args, self.mln))
        elif isinstance(ground_atom, Logic.GroundAtom):
            atom_str = str(ground_atom.ground_atom)
            args = ground_atom.args
        if atom_str not in self:
            return
        del self._evidence[atom_str]
        doms = self.mln.predicate(pred_name).arg_doms
        dont_remove = set()
        for atom, _ in self:
            _, pred_name_, args_ = self.mln.logic.parse_literal(atom)
            doms_ = self.mln.predicate(pred_name_).arg_doms
            for arg, arg_, dom, dom_ in zip(args, args_, doms, doms_):
                if arg == arg_ and dom == dom_:
                    dont_remove.add((dom, arg))
        for (dom, arg) in zip(doms, args):
            if (dom, arg) not in dont_remove:
                if arg in self._domains[dom]:
                    self._domains[dom].remove(arg)
                if not self.domain(dom):
                    del self._domains[dom]

    def retract_all(self, pred_name):
        """
        Retracts all evidence atoms of the given predicate name in this database.
        """
        for a, _ in dict(self._evidence).items():
            _, pred, _ = self.mln.logic.parse_literal(a)
            if pred == pred_name:
                del self[a]

    def remove_var(self, domain, value):
        for atom in list(self.evidence):
            _, pred_name, args = self.mln.logic.parse_literal(atom)
            for dom, val in zip(self.mln.predicate(pred_name).arg_doms, args):
                if dom == domain and val == value:
                    del self._evidence[atom]
            self.domains[domain].remove(value)

    def to_file(self, filename):
        """
        Writes this database into the file with the given filename.
        """
        f = open(filename, 'w+')
        self.write(f, color=False)
        f.close()

    def __iter__(self):
        for atom, truth, in self._evidence.items():
            yield atom, truth

    def __add__(self, other):
        return self.union(other, mln=self.mln)

    def __iadd__(self, other):
        return self.union(other, mln=self.mln)

    def __setitem__(self, atom, truth):
        self.add(atom, truth)

    def __getitem__(self, atom):
        return self.evidence.get(atom)

    def __lshift__(self, arg):
        if type(arg) is tuple:
            self.add(arg[0], float(arg[1]))
        elif isinstance(arg, str):
            self.add(arg)

    def __rshift__(self, atom):
        self.retract(atom)

    def __contains__(self, el):
        atom_str = self._atom_str(el)
        return atom_str in self._evidence

    def __delitem__(self, key):
        self.retract(key)

    def __len__(self):
        return len(self.evidence)

    def is_empty(self):
        """
        Returns True if there is an assertion for any ground atom in this
        database and False if the truth values all ground atoms are None
        AND all domains are empty.
        """
        return not any(map(lambda x: 0 <= x <= 1, self._evidence.values())) and \
            len(self.domains) == 0

    @staticmethod
    def write_dbs(dbs, stream=sys.stdout, color=None, bars=False):
        if color is None:
            if stream != sys.stdout:
                color = False
            else:
                color = True
        str_dbs = []
        for db in dbs:
            s = StringIO()
            db.write(s, color=color, bars=bars)
            str_dbs.append(s.getvalue())
            s.close()
        stream.write('---\n'.join(str_dbs))

    @staticmethod
    def load(mln, db_files, ignore_unknown_preds=False, db=None):
        """
        Reads one or multiple database files containing literals and/or domains.
        Returns one or multiple databases where domains is dictionary mapping
        domain names to lists of constants defined in the database
        and evidence is a dictionary mapping ground atom strings to truth values
        """
        # pdb.set_trace()
        if type(db_files) is not list:
            db_files = [db_files]
        dbs = []
        for db_path in db_files:
            if isinstance(db_path, str):
                db_path = MLNPath(db_path)
            if isinstance(db_path, MLNPath):
                project_path = None
                if db_path.project is not None:
                    project_path = db_path.projectloc
                dirs = [os.path.dirname(fp) for fp in db_files]
                dbs_ = parse_db(mln, content=db_path.content, ignore_unknown_preds=ignore_unknown_preds,
                                db=db, dirs=dirs, project_path=project_path)
                dbs.extend(dbs_)
        return dbs


def parse_db(mln, content, ignore_unknown_preds=False, db=None, dirs=['.'], project_path=None):
    """
    Reads one or more databases in a string representation and returns
    the respective Database objects.
    """
    log = logs.getlogger('db')
    content = strip_comments(content)
    allow_multiple = True
    if db is None:
        allow_multiple = True
        db = DataBase(mln, ignore_unknown_preds=ignore_unknown_preds)
    dbs = []
    for line, l in enumerate(content.split("\n")):
        l = l.strip()
        if l == '':
            continue
        elif l == '---' and not db.is_empty():
            dbs.append(db)
            db = DataBase(mln)
            continue
        elif "{" in l:
            dom_name, constants = db.mln.logic.parse_formula(l)
            dom_names = [dom_name for _ in constants]
        elif l.startswith('#include'):
            filename = l[len("#include "):].strip()
            m = re.match(r'"(?P<filename>.+)"', filename)
            if m is not None:
                filename = m.group('filename')
                # if the path is relative, look for the respective file
                # relatively to all paths specified. Take the first file matching.
                if not MLNPath(filename).exists:
                    include_filename = None
                    for d in dirs:
                        mlnp = '/'.join([d, filename])
                        if MLNPath(mlnp).exists:
                            include_filename = mlnp
                            break
                    if include_filename is None:
                        raise Exception('File not found: %s' % filename)
                else:
                    include_filename = filename
            else:
                m = re.match(r'<(?P<filename>.+)>', filename)
                if m is not None:
                    filename = m.group('filename')
                include_filename = ':'.join([project_path, filename])
            logger.debug('Including file: "%s"' % include_filename)
            p = MLNPath(include_filename)
            dbs.extend(parse_db(content=MLNPath(include_filename).content, ignore_unknown_preds=ignore_unknown_preds,
                                dirs=[p.resolve_path()] + dirs,
                                projectpath=ifnone(p.project, project_path, lambda x: '/'.join(p.path + [x])), mln=mln))
            # valued evidence
        elif l[0] in "0123456789":
            s = l.find(" ")
            ground_atom = l[s + 1:].replace(" ", "")
            value = float(l[:s])
            _, pred_name, constants = mln.logic.parse_literal(
                    ground_atom)  # TODO Should we allow soft evidence on non-atoms here? (This assumes atoms)
            dom_names = mln.predicate(pred_name).arg_doms
            db << (ground_atom, value)
            # literal
        else:
            true, pred_name, constants = mln.logic.parse_literal(l)
            if mln.predicate(pred_name) is None and ignore_unknown_preds:
                log.debug('Predicate "%s" is undefined.' % pred_name)
                continue
            dom_names = mln.predicate(pred_name).arg_doms
            # save evidence
            true = 1 if true else 0
            db << ("%s(%s)" % (pred_name, ",".join(constants)), true)

            # expand domains
        if len(dom_names) != len(constants):
            raise Exception("Ground atom %s in database %d has wrong number of parameters" % (l, len(dbs)))

        for i, c in enumerate(constants):
            db.domain({dom_names[i]: c})

        if not db.is_empty():
            dbs.append(db)
        return dbs

