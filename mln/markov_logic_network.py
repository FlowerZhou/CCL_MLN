"""
Markov Logic Network
"""
from typing import Iterator
import pdb
# from baize.logic import Predicate
# from baize.utils.vocab import Vocabulary
import pyparsing
from pyparsing import ParseException
from dnutils import logs, ifnone, out
import copy
import sys
import os
from logic.fol import FirstOrderLogic
import platform
from mln.method import InferenceMethods, LearningMethods
from mln.markov_random_field import MRF
import os
from mln.util import StopWatch, merge_dom, fstr, strip_comments
from mln.mlnpreds import Predicate
from mln.database import DataBase
from logic.elements import HARD
import sys
import re
import traceback
from mln.learning.bpll import BPLL
from mln.project import MLNPath
import logging
import torch
import torch.nn as nn

# from mln.parsers.antlr.mln_parsers import *


logger = logs.getlogger(__name__)


class MarkovLogicNetwork(object):
    """
    Represents a Markov logic network.

    :member formulas:    a list of :class:`logic.common.Formula` objects representing the formulas of the MLN.
    :member predicates:  a dict mapping predicate names to :class:`mlnpreds.Predicate` objects.

    :param logic:        (string) the type of logic to be used in this MLN. Possible values
                         are `FirstOrderLogic` and `FuzzyLogic`.
    :param grammar:      (string) the syntax to be used. Possible grammars are
                         `PRACGrammar` and `StandardGrammar`.
    :param mln_file:      can be a path to an MLN file or a file object.
    """

    def __init__(self, logic='FirstOrderLogic', grammar='StandardGrammar', mln_file=None):
        # instantiate the logic and grammar
        logic_str = '%s("%s", self)' % (logic, grammar)
        self.logic = eval(logic_str)
        logger.debug('Creating MLN with %s syntax and %s semantics' % (grammar, logic))
        self._predicates = {}  # maps from predicate name to the predicate instance
        self.domains = {}  # maps from domain names to list of values
        self._formulas = []  # list of MLNFormula instances
        self.domain_decls = []
        self.weights = []
        self.fix_weights = []
        self.vars = {}
        self._unique_templ_vars = []
        self._materialized = False
        if mln_file is not None:
            MarkovLogicNetwork.load(mln_file, logic=logic, grammar=grammar, mln=self)
            return
        self.closed_world_preds = []
        self._probreqs = []
        self.formula_groups = []
        self.watch = StopWatch()
        self.rule_weights_lin = None

    @property
    def predicates(self):
        return list(self.iter_predicates())

    @property
    def probreqs(self):
        return self._probreqs

    @property
    def formulas(self):
        return list(self._formulas)

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, wts):
        self._weights = wts

    @property
    def fix_weights(self):
        return self._fix_weights

    @fix_weights.setter
    def fix_weights(self, fw):
        self._fix_weights = fw

    @property
    def weighted_formulas(self):
        return [f for f in self._formulas if f.weight is not HARD]

    @property
    def predicate_name(self):  # predicate name
        return [p.name for p in self.predicates]

    def get_predicate(self, name):
        return self._predicates[name]

    def predicate(self, predicate):
        """
        Returns the predicate object with the given predicate name, or declares a new predicate.

        If predicate is a string, this method returns the predicate object
        assiciated to the given predicate name. If it is a predicate instance, it declares the
        new predicate in this MLN and returns the MLN instance. In the latter case, this is
        equivalent to `MLN.declare_predicate()`.

        :param predicate:    name of the predicate to be returned or a `Predicate` instance
                             specifying the predicate to be declared.
        :returns:            the Predicate object or None if there is no predicate with this name.
                             If a new predicate is declared, returns this MLN instance.

        :Example:

        >>> mln = MarkovLogicNetwork()
        >>> mln.predicate(Predicate(foo, [arg0, arg1]))
               .predicate(Predicate(bar, [arg1, arg2])) # this declares predicates foo and bar
        >>> mln.predicate('foo')
        <Predicate: foo(arg0,arg1)>
        """

        if isinstance(predicate, Predicate):
            return self.declare_predicate(predicate)
        elif isinstance(predicate, str):
            return self._predicates.get(predicate, None)
        elif isinstance(predicate, pyparsing.ParseResults):
            return predicate.asList()
        else:
            raise Exception('Illegal type of argument predicate: %s' % type(predicate))

    def iter_predicates(self):
        """
        Yields the predicates defined in this MLN alphabetically ordered.
        """
        for name in sorted(self._predicates):
            yield self.predicate(name)

    def update_predicates(self, mln):
        """
        Merges the predicate definitions of this MLN with the definitions
        of the given one.

        :param mln:     an instance of an MLN object.
        """
        for pred in mln.iter_predicates():
            self.declare_predicate(pred)

    def declare_predicate(self, predicate):
        """
        Adds a predicate declaration to the MLN:

        :param predicate:      an instance of a Predicate or one of its subclasses
                               specifying a predicate declaration.
        """
        pred = self._predicates.get(predicate.name)
        if pred is not None and pred != predicate:
            raise Exception('Contradictory predicate definitions: %s <--> %s' % (pred, predicate))
        else:
            self._predicates[predicate.name] = predicate
            for dom in predicate.arg_doms:
                if dom not in self.domains:
                    self.domains[dom] = []
        return self

    def formula(self, formula, weight=0., fix_weight=False, unique_templ_vars=None):
        """
        Adds a formula to this MLN. The respective domains of constants
        are updated, if necessary. If `formula` is an integer, returns the formula
        with the respective index or the formula object that has been created from
        formula. The formula will be automatically tied to this MLN.

        :param formula:             a `Logic.Formula` object or a formula string
        :param weight:              an optional weight. May be a mathematical expression
                                    as a string (e.g. log(0.1)), real-valued number
                                    or `mln.infty` to indicate a hard formula.
        :param fix_weight:           indicates whether or not the weight of this
                                    formula should be fixed during learning.
        :param unique_templ_vars:   specifies a list of template variables that will create
                                    only unique combinations of expanded formulas
        """
        if isinstance(formula, str):
            formula = self.logic.parse_formula(formula)
        elif type(formula) is int:
            return self._formulas[formula]
        constants = {}
        formula.var_doms(None, constants)
        for domain, constants in constants.items():
            for c in constants:
                self.constant(domain, c)
        formula.mln = self
        formula.index = len(self._formulas)
        self._formulas.append(formula)
        self.weights.append(weight)
        self.fix_weights.append(fix_weight)
        self._unique_templ_vars.append(list(unique_templ_vars) if unique_templ_vars is not None else [])
        return self._formulas[-1]

    def _remove_formulas(self):
        self._formulas = []
        self.weights = []
        self.fix_weights = []
        self._unique_templ_vars = []

    def iter_formulas(self):
        """
        Returns a generator yielding (idx, formula) tuples.
        """
        for i, f in enumerate(self._formulas):
            yield i, f

    def weight(self, idx, weight=None):
        """
        Returns or sets the weight of the formula with index `idx`.
        """
        if weight is not None:
            self.weights[idx] = weight
        else:
            return float(self.weights[idx])

    def __lshift__(self, _input):
        parse_mln(_input, '.', logic=None, grammar=None, mln=self)

    def copy(self):
        """
        Returns a deep copy of this MLN, which is not yet materialized.
        """
        mln_ = MarkovLogicNetwork(logic=self.logic.__class__.__name__, grammar=self.logic.grammar.__class__.__name__)
        for pred in self.iter_predicates():
            mln_.predicate(copy.copy(pred))
        mln_.domain_decls = list(self.domain_decls)
        for i, f in self.iter_formulas():
            mln_.formula(f.copy(mln=mln_), weight=self.weight(i), fix_weight=self.fix_weights[i],
                         unique_templ_vars=self._unique_templ_vars[i])
        mln_.domains = dict(self.domains)
        mln_.vars = dict(self.vars)
        return mln_

    def forward(self, neg_mask_ls_ls, latent_var_inds_ls_ls, observed_rule_cnts, posterior_prob, flat_list,
                observed_vars_ls_ls):
        """
        compute the MLN potential give the posterior probability of latent variables
        :params neg_mask_ls_ls
        :params posterior prob_ls_ls
        :return

        weight wf can be viewed as the confidence score of the formula f
        """
        # pdb.set_trace()
        scores = torch.zeros(len(self._formulas), dtype=torch.float)
        for i in range(len(neg_mask_ls_ls)):
            neg_mask_ls = neg_mask_ls_ls[i]
            latent_var_inds_ls = latent_var_inds_ls_ls[i]
            observed_vars_ls = observed_vars_ls_ls[i]

        # sum of scores from gnd rules with latent vars
        for j in range(len(neg_mask_ls)):
            latent_neg_mask, observed_neg_mask = neg_mask_ls[j]
            latent_var_inds = latent_var_inds_ls[j]
            observed_vars = observed_vars_ls[j]
            z_probs = posterior_prob[latent_var_inds].unsqueeze(0)
            z_probs = torch.cat([1-z_probs, z_probs], dim=0)
            cartesian_prod = z_probs[:, 0]
            for j in range(1, z_probs.shape[1]):
                cartesian_prod = torch.ger(cartesian_prod, z_probs[:, j])
                cartesian_prod = cartesian_prod.view(-1)

            view_ls = [2 for _ in range(len(latent_neg_mask))]
            cartesian_prod = cartesian_prod.view(*[view_ls])
            # pdb.set_trace()
            if sum(observed_neg_mask) == 0:
                cartesian_prod[tuple(latent_neg_mask)] = 0.0
            scores[i] += cartesian_prod.sum()
        # pdb.set_trace()
        return self.rule_weights_lin(scores)

    def materialize(self, *dbs):
        """
        Materializes this MLN with respect to the databases given. This must
        be called before learning or inference can take place.

        Returns a new MLN instance containing expanded formula templates and
        materialized weights. Normally, this method should not be called from the outside.
        Also takes into account whether or not particular domain values or predicates
        are actually used in the data, i.e. if a predicate is not used in any
        of the databases, all formulas that make use of this predicate are ignored.
        :param dbs:     list of :class:`database.Database` objects for materialization.
        """
        logger.debug("materializing formula templates...")

        # obtain full domain with all objects
        full_domain = merge_dom(self.domains, *[db.domains for db in dbs])
        logger.debug('full domains: %s' % full_domain)
        mln_ = self.copy()
        # collect the admissible formula templates. templates might be not
        # admissible since the domain of a template variable might be empty.
        for ft in list(mln_.formulas):
            dom_names = ft.var_doms().values()
            if any([dom_name not in full_domain for dom_name in dom_names]):
                logger.debug('Discarding formula template %s, since it cannot be grounded (domain(s) %s empty).' % \
                             (fstr(ft), ','.join([d for d in dom_names if d not in full_domain])))
                mln_.rmf(ft)
        # collect the admissible predicates. a predicate may become inadmissible
        # if either the domain of one of its arguments is empty or there is
        # no formula containing the respective predicate.
        predicates_used = set()
        for _, f in mln_.iter_formulas():
            predicates_used.update(f.pred_names())
        for predicate in self.iter_predicates():
            remove = False
            if any([dom not in full_domain for dom in predicate.arg_doms]):
                logger.debug('Discarding predicate %s, since it cannot be grounded.' % predicate.name)
                remove = True
            if predicate.name not in predicates_used:
                logger.debug('Discarding predicate %s, since it is unused.' % predicate.name)
                remove = True
            if remove:
                del mln_._predicates[predicate.name]
        # permanently transfer domains of variables that were expanded from templates
        for _, ft in mln_.iter_formulas():
            dom_names = list(ft.template_variables().values())
            for dom_name in dom_names:
                mln_.domains[dom_name] = full_domain[dom_name]
        # materialize the formula templates
        mln__ = mln_.copy()
        mln__._remove_formulas()
        # pdb.set_trace()
        for i, template in mln_.iter_formulas():
            for variant in template.template_variants():
                idx = len(mln__._formulas)
                f = mln__.formula(variant, weight=template.weight, fix_weight=mln_.fix_weights[i])
                f.index = idx
        mln__._materialized = True
        return mln__

    def constant(self, domain, *values):
        """
        Adds to the MLN a constant domain value to the domain specified.

        If the domain doesn't exist, it is created.

        :param domain:    (string) the name of the domain the given value shall be added to.
        :param values:     (string) the values to be added.
        """
        if domain not in self.domains:
            self.domains[domain] = []
        dom = self.domains[domain]
        for value in values:
            if value not in dom:
                dom.append(value)
        return self

    def ground(self, db):
        """
        Creates and returns a ground Markov Random Field for the given database.

        :param db:         database filename (string) or Database object

        """
        logger.debug('creating ground MRF...')
        mrf = MRF(self, db)
        # pdb.set_trace()
        for pred in self.predicates:
            for ground_atom in pred.ground_atoms(self, mrf.domains):
                mrf.ground_atom(ground_atom.pred_name, *ground_atom.args)
        # pdb.set_trace()
        evidence = dict([(atom, value) for atom, value in db.evidence.items() if mrf.ground_atom(atom) is not None])
        mrf.set_evidence(evidence, erase=False)
        return mrf

    def update_domain(self, domain):
        """
        Combines the existing domain (if any) with the given one.

        :param domain: a dictionary with domain Name to list of string constants to add
        """
        for dom_name in domain:
            break
        for value in domain[dom_name]:
            self.constant(dom_name, value)

    @staticmethod
    def load(files, logic='FirstOrderLogic', grammar='StandardGrammar', mln=None):
        """
        Reads an MLN object from a file or a set of files.

        :param files:     one or more MLNPath strings. If multiple file names are given,
                          the contents of all files will be concatenated.
        :param logic:     (string) the type of logic to be used.
        :param grammar:   (string) the syntax to be used for parsing the MLN file.
        """
        # read MLN file
        # pdb.set_trace()
        text = ''
        if files is not None:
            if not type(files) is list:
                files = [files]
            project_path = None
            for f in files:
                if isinstance(f, str):
                    p = MLNPath(f)
                    if p.project is not None:
                        project_path = p.projectloc
                    text += p.content
                elif isinstance(f, MLNPath):
                    text += f.content
                else:
                    raise Exception('Unexpected file specification: %s' % str(f))
            dirs = [os.path.dirname(fn) for fn in files]
            return parse_mln(text, searchpaths=dirs, projectpath=project_path, logic=logic, grammar=grammar, mln=mln)
        raise Exception('No mln files given.')

    def to_file(self, filename):
        """
        Creates the file with the given filename and writes this MLN into it.
        """
        f = open(filename, 'w+')
        self.write(f, color=False)
        f.close()

    def write(self, stream=sys.stdout, color=None):
        """
        Writes the MLN to the given stream.

        The default stream is `sys.stdout`. In order to print the MLN to the console, a simple
        call of `mln.write()` is sufficient. If color is not specified (is None), then the
        output to the console will be colored and uncolored for every other stream.

        :param stream:        the stream to write the MLN to.
        :param color:         whether or not output should be colorized.
        """
        if color is None:
            if stream != sys.stdout:
                color = False
            else:
                color = True
        if 'learnwts_message' in dir(self):
            stream.write("/*\n%s*/\n\n" % self.learnwts_message)
            # domain declarations
        if self.domain_decls:
            stream.write("// domain declarations\n")
        for d in self.domain_decls:
            stream.write("%s\n" % d)
        stream.write('\n')
        # variable definitions
        if self.vars:
            stream.write('// variable definitions\n')
        for var, val in self.vars.items():
            stream.write('%s = %s' % (var, val))
        stream.write('\n')
        stream.write("\n// predicate declarations\n")
        for predicate in self.iter_predicates():
            stream.write("%s(%s)\n" % (predicate.name, predicate.arg_str()))
        stream.write("\n// formulas\n")
        for idx, formula in self.iter_formulas():
            if self._unique_templ_vars[idx]:
                stream.write('#unique{%s}\n' % ','.join(self._unique_templ_vars[idx]))
            if formula.weight == HARD:
                stream.write("%s.\n" % fstr(formula.cstr(color)))
            else:
                try:
                    w = "%-10.6f" % float(eval(str(formula.weight)))
                except:
                    w = str(formula.weight)
                stream.write("%s  %s\n" % (w, fstr(formula)))

    def learn(self, databases, method=BPLL, **params):
        """
        Triggers the learning parameter learning process for a given set of databases.
        Returns a new MLN object with the learned parameters.

        :param databases: list of :class:`mln.database.Database` objects or filenames
        """
        verbose = params.get('verbose', False)

        # get a list of database objects
        if not databases:
            raise Exception('At least one database is needed for learning.')
        dbs = []
        for db in databases:
            if isinstance(db, str):
                db = DataBase.load(self, db)
                if type(db) is list:
                    dbs.extend(db)
                else:
                    dbs.append(db)
            elif type(db) is list:
                dbs.extend(db)
            else:
                dbs.append(db)
        logger.debug('loaded %s evidence databases for learning' % len(dbs))
        new_mln = self.materialize(*dbs)
        # pdb.set_trace()
        logger.debug('MLN predicates:')
        for p in new_mln.predicates:
            logger.debug(p)
        logger.debug('MLN domains:')
        for d in new_mln.domains.items():
            logger.debug(d)
        if not new_mln.formulas:
            raise Exception('No formulas in the materialized MLN.')
        logger.debug('MLN formulas:')
        for f in new_mln.formulas:
            logger.debug('%s %s' % (str(f.weight).ljust(10, ' '), f))
        # run learner
        if len(dbs) == 1:
            mrf = new_mln.ground(dbs[0])
            logger.debug('Loading %s-Learner' % method.__name__)
            learner = method(mrf, **params)
        if verbose:
            "learner: %s" % learner.name
        # pdb.set_trace()
        # mrf.reduce_variables(mrf.variables)
        wt = learner.run(**params)
        new_mln.weights = wt
        new_mln.rule_weights_lin = nn.Linear(len(self._formulas), 1, bias=False)
        # pdb.set_trace()
        new_mln.rule_weights_lin.weight = nn.Parameter(
            torch.tensor([[int(float(weight)) for weight in new_mln.weights]],
                         dtype=torch.float))
        # pdb.set_trace()
        if params.get('ignore_zero_weight_formulas', False):
            formulas = list(new_mln.formulas)
            weights = list(new_mln.weights)
            fix = list(new_mln.fix_weights)
            new_mln._remove_formulas()
            for f, w, fi in zip(formulas, weights, fix):
                if w != 0:
                    new_mln.formula(f, w, fi)
        return new_mln


def parse_mln(text, searchpaths=['.'], projectpath=None, logic='FirstOrderLogic', grammar='StandardGrammar', mln=None):

    formulatemplates = []
    text = str(text)
    text = strip_comments(text)
    if mln is None:
        mln = MarkovLogicNetwork(logic, grammar)
    # read lines
    mln.hard_formulas = []
    inGroup = False
    idxGroup = -1
    fuzzy = False
    pseudofuzzy = False
    uniquevars = None

    lines = text.split("\n")
    iLine = 0
    while iLine < len(lines):
        line = lines[iLine]
        iLine += 1
        line = line.strip()
        try:
            if len(line) == 0:
                continue
            if '=' in line:
                # try normal domain definition
                parse = mln.logic.parse_domain(line)
                if parse is not None:
                    dom_name, constants = parse
                    dom_name = str(dom_name)
                    constants = list(map(str, constants))
                    if dom_name in mln.domains:
                        logger.debug("Domain redefinition: Domain '%s' is being updated with values %s." %
                                     (dom_name, str(constants)))
                    if dom_name not in mln.domains:
                        mln.domains[dom_name] = []
                    mln.constant(dom_name, *constants)
                    mln.domain_decls.append(line)
                    continue

            # variable definition
            if re.match(r'(\$\w+)\s*=(.+)', line):
                m = re.match(r'(\$\w+)\s*=(.+)', line)
                if m is None:
                    raise Exception("Variable assigment malformed: %s" % line)
                mln.vars[m.group(1)] = "%s" % m.group(2).strip()
                continue
                # predicate decl or formula with weight
            else:
                isHard = False
                isPredDecl = False
                if line[-1] == '.':  # hard (without explicit weight -> determine later)
                    isHard = True
                    formula = line[:-1]
                else:  # with weight
                    # try predicate declaration
                    isPredDecl = True
                    try:
                        pred = mln.logic.parse_predicate(line)
                    except Exception as e:
                        isPredDecl = False
                if isPredDecl:
                    predname = str(pred[0])
                    argdoms = list(map(str, pred[1]))
                    softmutex = False
                    mutex = None
                    for i, dom in enumerate(argdoms):
                        if dom[-1] in ('!', '?'):
                            if mutex is not None:
                                raise Exception('More than one arguments are specified as (soft-)functional')
                            if fuzzy:
                                raise Exception('(Soft-)functional predicates must not be fuzzy.')
                            mutex = i
                        if dom[-1] == '?':
                            softmutex = True
                    argdoms = [x.strip('!?') for x in argdoms]
                    pred = None
                    pred = Predicate(predname, argdoms)

                    mln.predicate(pred)
                    continue
                else:
                    # formula (template) with weight or terminated by '.'
                    if not isHard:
                        spacepos = line.find(' ')
                        weight = line[:spacepos]
                        formula = line[spacepos:].strip()
                    try:
                        formula = mln.logic.parse_formula(formula)
                        # pdb.set_trace()
                        if isHard:
                            weight = HARD  # not set until instantiation when other weights are known

                        fix_weight = False
                        # expand predicate groups
                        for variant in formula.expand_group_lists():
                            mln.formula(variant, weight=weight, fix_weight=fix_weight, unique_templ_vars=uniquevars)
                        if uniquevars:
                            uniquevars = None
                    except ParseException as e:
                        raise Exception("Error parsing formula '%s'\n" % formula)
                if fuzzy and not isPredDecl:
                    raise Exception('"#fuzzy" decorator not allowed at this place: %s' % line)
        except Exception as err:
            sys.stderr.write("Error processing line '%s'\n" % line)
            cls, e, tb = sys.exc_info()
            traceback.print_tb(tb)
            raise Exception(err)

    # augment domains with constants appearing in formula templates
    for _, f in mln.iter_formulas():
        constants = {}
        f.var_doms(None, constants)
        for domain, constants in constants.items():
            for c in constants:
                mln.constant(domain, c)
    return mln

