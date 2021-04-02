"""
logic system
"""
from typing import List
from mln.constants import auto, HARD, inherit
from collections import defaultdict
from mln.util import dict_union
import logging
import itertools
import pdb
from .grammar import StandardGrammar
from functools import reduce
from dnutils import logs, ifnone

logger = logging.getLogger(__name__)


class PredicateArgument(object):

    def __init__(self, type_=None, name=None, unique=False):
        self.type_ = type_
        self.name = name
        self.unique = unique


class Logic(object):
    """
    Abstract factory class for instantiating logical constructs like conjunctions,
    disjunctions etc. Every specific logic should implement the methods and return
    an instance of the respective element. They also might override the respective
    implementations and behavior of the logic.
    """
    def __init__(self, grammar, mln):
        if grammar not in ('StandardGrammar', 'PRACGrammar'):
            raise Exception('Invalid grammar: %s' % grammar)
        print(grammar)
        self.grammar = eval(grammar)(self)
        self.mln = mln

    # make the class object pickle-able
    def __getstate__(self):
        d = self.__dict__.copy()
        d['grammar'] = type(self.grammar).__name__
        return d

    # make the class object unpickle-able
    def __setstate__(self, d):
        self.__dict__ = d
        self.grammar = eval(d['grammar'])(self)

    class Constraints(object):
        """
        Super class of every constraint.
        """
        def template_variants(self, mln):
            raise Exception("%s does not implement get_template_variants" % str(type(self)))

        def truth(self, world):
            raise Exception("%s does not implement truth" % str(type(self)))

        def islogical(self):
            """
            Returns whether this is a logical constraint, i.e. a logical formula
            """
            raise Exception("%s does not implement islogical" % str(type(self)))

        def iter_groundings(self, mrf, simplify=False, domains=None):
            """
            Iteratively yields the groundings of the formula for the given ground MRF
            - simplify:     If set to True, the grounded formulas will be simplified
                            according to the evidence set in the MRF.
            - domains:      If None, the default domains will be used for grounding.
                            If its a dict mapping the variable names to a list of values,
                            these values will be used instead.
            """
            raise Exception("%s does not implement iter_groundings" % str(type(self)))

        def index_ground_atoms(self, l=None):
            raise Exception("%s does not implement index_ground_atoms" % str(type(self)))

        def ground_atoms(self, l=None):
            raise Exception("%s does not implement ground_atoms" % str(type(self)))

    class Formula(Constraints):
        """
        The base class for all logical formulas.
        """
        def __init__(self, mln=None, index=None):
            # print("Enter Formula")
            self.mln = mln
            if index == auto and mln is not None:
                self.index = len(mln.formulas)
            else:
                self.index = index

        @property
        def index(self):
            """
            the weight of formula
            """
            return self._index

        @index.setter
        def index(self, index):
            self._index = index

        @property
        def mln(self):
            return self._mln

        @mln.setter
        def mln(self, mln):
            if hasattr(self, 'children'):
                for child in self.children:
                    child.mln = mln
            self._mln = mln

        @property
        def weight(self):
            return self.mln.weight(self.index)

        @weight.setter
        def weight(self, w):
            if self.index is None:
                raise Exception('%s does not have an index' % str(self))
            self.mln.weight(self.index, w)

        @property
        def is_hard(self):
            return self.weight == HARD

        def contain_ground_atom(self, ground_atom_index):
            if not hasattr(self, "children"):
                return False
            for child in self.children:
                if child.contain_ground_atom(ground_atom_index):
                    return True
            return False

        def ground_atom_indices(self, l=None):
            if l is None:
                l = []
            if not hasattr(self, "children"):
                return l
            for child in self.children:
                child.ground_atom_indices(l)
            return l

        def ground_atoms(self, l=None):
            if l is None:
                l = []
            if not hasattr(self, "children"):
                return l
            for child in self.children:
                child.ground_atoms(l)
            return l

        def template_atom(self):
            """
             Returns a list of template variants of all atoms
            that can be generated from this formula and the given mln.

            :Example:

            foo(?x, +?y) ^ bar(?x, +?z) --> [foo(?x, X1), foo(?x, X2), ...,
                                                      bar(?x, Z1), bar(?x, Z2), ...]
            """
            temp_atom = []
            for literal in self.literals():
                for temp in literal.template_variants():
                    temp_atom.append(temp)
            return temp_atom

        def atomic_constituents(self, of_type=None):
            const = list(self.literals())
            if of_type is None:
                return const
            else:
                return filter(lambda c: isinstance(c, of_type), const)

        def constituent(self, of_type=None, const=None):
            """
            Returns a list that contains all constituents (atomic and non-atomic)
            of this formula, optionally filtered by ``of_type``.
            :param of_type:
            :param const
            """
            if const is None:
                const = []
            if of_type is None or type(self) is of_type:
                const.append(self)
            if hasattr(self, "children"):
                for child in self.children:
                    child.constituent(of_type, const)
            return const

        def template_variants(self):
            """
            Gets all the template variants of the formula for the given MLN
            """
            unique_var = list(self.mln._unique_templ_vars[self.index])
            vardoms = self.template_variables()
            unique_var_ = defaultdict(set)
            for var in unique_var:
                dom = vardoms[var]
                unique_var_[dom].add(var)
            assignments = []
            for domain, variable in unique_var_.items():
                group = []
                dom_values = self.mln.domains[domain]
                if not dom_values:
                    logger.warning(
                        'Template variants cannot be constructed since the domain "{}" is empty.'.format(domain))
                for values in itertools.combinations(dom_values, len(variable)):
                    group.append(dict([(var, val) for var, val in zip(variable, values)]))
                assignments.append(group)
            for domain, variable in vardoms.items():
                if variable in unique_var:
                    continue
                group = []
                dom_values = self.mln.domains[domain]
                if not dom_values:
                    logger.warning(
                        'Template variants cannot be constructed since the domain "{}" is empty.'.format(domain))
                for value in self.mln.domains[domain]:
                    group.append(dict([(variable, value)]))
                assignments.append(group)

            def product(assign, result=[]):
                if len(assign) == 0:
                    yield result
                    return
                for a in assign[0]:
                    for r in product(assign[1:], result + [a]):
                        yield r

            for assignment in product(assignments):
                if assignment:
                    for t in self._ground_template(reduce(lambda x, y: dict_union(x, y),
                                                          itertools.chain(assignment))):
                        yield t
                else:
                    for t in self._ground_template({}):
                        yield t

        def _ground_template(self, assignment):
            """
            Grounds this formula for the given assignment of template variables
            and returns a list of formulas, the list of template variants
            - assignment: a mapping from variable names to constants
            """
            raise Exception("%s does not implement _ground_template" % str(type(self)))

        def template_variables(self):
            raise Exception("%s does not implement template_variables" % str(type(self)))

        def iter_var_groundings(self, mrf, partial=None):
            """
            Yields dictionaries mapping variable names to values
            this formula may be grounded with without grounding it. If there are not free
            variables in the formula, returns an empty dict.
            """
            variables = self.var_doms()
            if partial is not None:
                for v in [p for p in partial if p in variables]:
                    del variables[v]
            for assignment in self._iter_var_grounding(mrf, variables, {}):
                yield assignment

        def _iter_var_grounding(self, mrf, variables, assignment):
            """
            if all variables has been assigned a value
            """
            if variables == {}:
                yield assignment
                return
            variables = dict(variables)
            var_name, dom_name = variables.popitem()
            domain = mrf.domains[dom_name]
            assignment = dict(assignment)
            for value in domain:
                assignment[var_name] = value
                for assign in self._iter_var_grounding(mrf, dict(variables), assignment):
                    yield assign

        def iter_groundings(self, mrf, simplify=False, domains=None):
            """
            Iteratively yields the groundings of the formula for the given grounder

            :param mrf:          an object, such as an MRF instance
            :param simplify:     If set to True, the grounded formulas will be simplified
                                 according to the evidence set in the MRF.
            :param domains:      If None, the default domains will be used for grounding.
                                 If its a dict mapping the variable names to a list of values,
                                 these values will be used instead.
            :returns:            a generator for all ground formulas
            """
            variables = self.var_doms()
            for grounding in self._iter_groundings(mrf, variables, {}, simplify, domains):
                yield grounding

        def _iter_groundings(self, mrf, variables, assignment, simplify=False, domains=None):
            """
            if all variables have been grounded
            """
            if not variables:
                gt = self.ground(mrf, assignment, simplify, domains)
                yield gt
                return
            var_name, dom_name = variables.popitem()
            domain = domains[var_name] if domains is not None else mrf.domains[dom_name]
            for value in domain:
                assignment[var_name] = value
                for gf in self._iter_groundings(mrf, dict(variables), assignment, simplify, domains):
                    yield gf

        def iter_true_var_assignments(self, mrf, world=None, truth_thr=1.0, strict=False, unknown=False, partial=None):
            """
            Iteratively yields the variable assignments (as a dict) for which this
            formula exceeds the given truth threshold.

            Same as iter_groundings, but returns variable mappings only for assignments rendering this formula true.

            :param mrf:        the MRF instance to be used for the grounding.
            :param world:      the possible world values. if `None`, the evidence in the MRF is used.
            :param truth_thr:        a truth threshold for this formula. Only variable assignments rendering this
                               formula true with at least this truth value will be returned.
            :param strict:     if `True`, the truth value of the formula must be strictly greater than the `thr`.
                               if `False`, it can be greater or equal.
            :param unknown:    If `True`, also groundings with the truth value `None` are returned

            """
            if world is None:
                world = list(mrf.evidence)
            if partial is None:
                partial = {}
            variables = self.var_doms()
            for var in partial:
                if var in variables:
                    del variables[var]
            for assignment in self._iter_true_var_assignments(mrf, variables, partial, world, dict(variables),
                                                              truth_thr=truth_thr, strict=strict, unknown=unknown):
                yield assignment

        def _iter_true_var_assignments(self, mrf, variables, assignment, world, all_vars, truth_thr=1.0,
                                       strict=False, unknown=False):
            """
            if all variables have been grounded
            """
            if variables == {}:
                gf = self.ground(mrf, assignment)
                truth = gf(world)
                if ((((truth >= truth_thr) if not strict else (truth > truth_thr)) and truth is not None)
                or (truth is None and unknown)):
                    true_assignment = {}
                    for v in all_vars:
                        true_assignment[v] = assignment[v]
                    yield true_assignment
                return
            var_name, dom_name = variables.popitem()
            assignment_ = dict(assignment)
            for value in mrf.domains[dom_name]:
                assignment_[var_name] = value
                for ass in self._iter_true_var_assignments(mrf, dict(variables), assignment_, world,
                                                           all_vars, truth_thr=truth_thr, strict=strict,
                                                           unknown=unknown):
                    yield ass

        def var_doms(self, variables=None, constants=None):
            """
            Returns a dictionary mapping each variable name in this formula to
            its domain name as specified in the associated MLN.
            """
            raise Exception("%s does not implement var_doms()" % str(type(self)))

        def pred_names(self, pred_names=None):
            """
            Returns a list of all predicate names used in this formula.
            """
            raise Exception('%s does not implement prednames()' % str(type(self)))

        def ground(self, mrf, assignment, simplify=False, partial=False):
            """
            Grounds the formula using the given assignment of variables to values/constants and,
            if given a list in referencedAtoms, fills that list with indices of ground atoms
            that the resulting ground formula uses

            :param mrf:            the :class:`mln.base.MRF` instance
            :param assignment:     mapping of variable names to values
            :param simplify:       whether or not the formula shall be simplified wrt
            :param partial:        by default, only complete groundings are allowed. If `partial` is `True`,
                                   the result formula may also contain free variables.
            :returns:              a new formula object instance representing the grounded formula
            """
            raise Exception("%s does not implement ground" % str(type(self)))

        def copy(self, mln=None, index=inherit):
            """
            Produces a deep copy of this formula.

            If `mln` is specified, the copied formula will be tied to `mln`. If not, it will be
            tied to the same MLN as the original formula is. If `index` is None, the index of the
            original formula will be used.
            """
            raise Exception('%s does not implement copy()' % str(type(self)))

        def var_dom(self, var_name):
            """
            Returns the domain values of the variable with name `var_dom`.
            """
            return self.mln.domains.get(self.var_doms()[var_name])

        def cnf(self, level=0):
            """
            Convert to conjunctive normal form.
            """
            return self

        def nnf(self, level=0):
            """
            convert to negation normal form
            """
            return self.copy()

        def islogical(self):
            return True

        def simplify(self, mrf):
            """
            Simplify the formula by evaluating it with respect to the ground atoms given
            by the evidence in the mrf.
            """
            raise Exception('%s does not implement simplify()' % str(type(self)))

        def literals(self):
            """
            Traverses the formula and returns a generator for the literals it contains.
            """
            if not hasattr(self, 'children'):
                yield self
                return
            else:
                for child in self.children:
                    for lit in child.literals():
                        yield lit

        def expand_group_lists(self):
            """
            return lists of formulas
            """
            for t in self._ground_template({}):
                yield t

        def truth(self, world):
            """
            Evaluates the formula for its truth wrt. the truth values
            of ground atoms in the possible world `world`.

            :param world:     a vector of truth values representing a possible world.
            :returns:         the truth of the formula in `world` in [0,1] or None if
                              the truth value cannot be determined.
            """
            raise Exception('%s does not implement truth()' % str(type(self)))

        def count_groundings(self, mrf):
            gf_count = 1
            for _, dom in self.var_doms().items():
                domain = mrf.domains[dom]
                gf_count += len(domain)
            return gf_count

        def max_truth(self, world):
            """
            Returns the maximum truth value of this formula given the evidence.
            For FOL, this is always 1 if the formula is not rendered false by evidence.
            """
            raise Exception('%s does not implement maxtruth()' % self.__class__.__name__)

        def min_truth(self, world):
            raise Exception('%s does not implement mintruth()' % self.__class__.__name__)

        def __call__(self, world):
            return self.truth(world)

        def __repr__(self):
            return '<%s: %s>' % (self.__class__.__name__, str(self))

    class ComplexFormula(Formula):
        """
        A formula has other child-formulas
        """

        def __init__(self, mln, index=None):
            Formula.__init__(self, mln, index)

        def var_doms(self, variables=None, constants=None):
            """
            Get the free (unquantified) variables of the formula in a dict that maps the variable name
            to the correspond domain name.
            The vars and constants parameters can be omitted.
            If vars is given, it must be a dictionary with already known variables.
            If constants is given, then it must be a dictionary that is to be extended with
            all constants appearing in the formula;
            it will be a dictionary mapping domain names to lists of constants
            If constants is not given, then constants are not collected, only variables.
            The dictionary of variables is returned.
            """
            if variables is None:
                variables = defaultdict(set)
            for child in self.children:
                if not hasattr(child, 'var_doms'):
                    continue
                variables = child.var_doms(variables, constants)
            return variables

        def constants(self, constants=None):
            """
            Get the constants appearing in the formula in a dict that maps the constant
            name to the domain name the constant belongs to.
            """
            if constants is None:
                constants = defaultdict
            for child in self.children:
                if not hasattr(child, "constants"):
                    continue
                constants = child.constants(constants)
            return constants

        def ground(self, mrf, assignment, simplify=False, partial=False):
            """
            return True or False
            """
            children = []
            for child in self.children:
                ground_child = child.ground(mrf, assignment, simplify, partial)
                children.append(ground_child)  # if simplify, add [<TrueFalse: True/False>] else add [ground_literal]
            # if simplify, disjunction: True v False
            ground_formula = self.mln.logic.create(type(self), children, mln=self.mln, index=self.index)
            if simplify:
                ground_formula = ground_formula.simplify(mrf.evidence)   # return True or False
            ground_formula.index = self.index
            return ground_formula

        def _ground_template(self, assignment):
            # pdb.set_trace()
            variants = [[]]
            for child in self.children:
                child_variants = child._ground_template(assignment)
                new_variants = []
                for variant in variants:
                    for child_variant in child_variants:
                        v = list(variant)
                        v.append(child_variant)
                        new_variants.append(v)
                variants = new_variants
            final_variants = []
            for variant in variants:
                if isinstance(self, Logic.Exist):
                    final_variants.append(self.mln.logic.exist(self.vars, variant[0], mln=self.mln))
                else:
                    final_variants.append(self.mln.logic.create(type(self), variant, mln=self.mln))
            return final_variants

        def template_variables(self, variables=None):
            if variables is None:
                variables = {}
            for child in self.children:
                child.template_variables(variables)
            return variables

        def pred_names(self, pred_names=None):
            if pred_names is None:
                pred_names = []
            for child in self.children:
                if not hasattr(child, 'pred_names'):
                    continue
                pred_names = child.pred_names(pred_names)
            return pred_names

        def copy(self, mln=None, index=inherit):
            children = []
            for child in self.children:
                child_ = child.copy(mln=ifnone(mln, self.mln), index=None)
                children.append(child_)
            return type(self)(children, mln=ifnone(mln, self.mln), index=self.index if index is inherit else index)

    class Conjunction(ComplexFormula):
        """
        represent logic conjunction
        """
        def __init__(self, children, mln, index=None):
            Logic.Formula.__init__(self, mln, index)
            self.children = children

        @property
        def children(self):
            return self._children

        @children.setter
        def children(self, children):
            if len(children) < 2:
                raise Exception('Conjunction needs at least 2 children.')
            self._children = children

        def __str__(self):
            return ' ^ '.join(
                map(lambda c: ('(%s)' % str(c)) if isinstance(c, Logic.ComplexFormula) else str(c), self.children))

        def cstr(self, color=False):
            return ' ^ '.join(
                map(lambda c: ('(%s)' % c.cstr(color)) if isinstance(c, Logic.ComplexFormula) else c.cstr(color),
                    self.children))

        def latex(self):
            return ' \land '.join(
                map(lambda c: ('(%s)' % c.latex()) if isinstance(c, Logic.ComplexFormula) else c.latex(),
                    self.children))

        def max_truth(self, world):  # may have some problems
            min_truth = 1
            for c in self.children:
                truth = c.truth(world)
                if truth is None:
                    continue
                if truth < min_truth:
                    min_truth = truth
            return min_truth

        def min_truth(self, world):
            pdb.set_trace()
            min_truth = 1
            for c in self.children:
                truth = c.truth(world)
                if truth is None:
                    return 0
                if truth < min_truth:
                    min_truth = truth
            return min_truth

        def cnf(self, level=0):
            clauses = []
            lit_sets = []
            for child in self.children:
                c = child.cnf(level+1)
                if isinstance(c, Logic.Conjunction):
                    l = c.children
                else:
                    l = [c]
                for clause in l: # (clause is either a disjunction, a literal or a constant)
                    if isinstance(self, Logic.TrueFalse):
                        if clause.truth() == 1:
                            continue
                        elif clause.truth() == 0:
                            return self.mln.logic.true_false(0, mln=self.mln, index=self.index)
                    if hasattr(clause, "children"):
                        lit_set = set(map(str, clause.children))
                    else:
                        lit_set = set([str(clause)])
                    do_add = True
                    i = 0
                    while i < len(lit_sets):
                        s = lit_sets[i]
                        if len(lit_set) < len(s):
                            if lit_set.issubset(s):
                                del lit_sets[i]
                                del clauses[i]
                                continue
                        else:
                            if lit_set.issuperset(s):
                                do_add = False
                                break
                        i += 1
                    if do_add:
                        clauses.append(clause)
                        lit_sets.append(lit_set)
            if not clauses:
                return self.mln.logic.true_false(1, mln=self.mln, index=self.index)
            elif len(clauses) == 1:
                return clauses[0].copy(index=self.index)
            return self.mln.logic.conjunction(clauses, mln=self.mln, index=self.index)

        def nnf(self, level=0):
            conjuncts = []
            for child in self.children:
                c = child.nnf(level+1)
                if isinstance(c, Logic.Conjunction):
                    conjuncts.extend(c.children)
                else:
                    conjuncts.append(c)
            return self.mln.logic.conjunction(conjuncts, mln=self.mln, index=self.index)

    class Disjunction(ComplexFormula):
        """
        represent a disjunction of formula
        """

        def __init__(self, children, mln, index=None):
            Formula.__init__(self, mln, index)
            self.children = children

        @property
        def children(self):
            return self._children

        @children.setter
        def children(self, children):
            if len(children) < 2:
                raise Exception('Disjunction needs at least 2 children.')
            self._children = children

        def __str__(self):
            return ' v '.join(
                map(lambda c: ('(%s)' % str(c)) if isinstance(c, Logic.ComplexFormula) else str(c), self.children))

        def cstr(self, color=False):
            return ' v '.join(
                map(lambda c: ('(%s)' % c.cstr(color)) if isinstance(c, Logic.ComplexFormula) else c.cstr(color),
                    self.children))

        def latex(self):
            return ' \lor '.join(
                map(lambda c: ('(%s)' % c.latex()) if isinstance(c, Logic.ComplexFormula) else c.latex(),
                    self.children))

        def max_truth(self, world):
            max_truth = 0
            for c in self.children:
                truth = c.truth(world)
                if truth is None:
                    return 1
                if truth > max_truth:
                    max_truth = truth
            return max_truth

        def min_truth(self, world):  # may have some problems
            max_truth = 0;
            for c in self.children:
                truth = c.truth(world)
                if truth is None:
                    continue
                if truth > max_truth:
                    max_truth = truth
            return max_truth

        def cnf(self, level=0):
            disj = []
            conj = []
            for child in self.children:
                c = child.cnf(level+1)    # convert child to CNF
                if isinstance(c, Logic.Conjunction):
                    conj.append(c)
                else:
                    if isinstance(c, Logic.Disjunction):
                        lits = c.children
                    else:
                        lits = [c]
                    for l in lits:
                        if isinstance(l, Logic.TrueFalse):
                            if l.truth():
                                return self.mln.logic.true_false(1, mln=self.mln, index=self.index)
                            else:
                                continue
                        l_ = l.copy()
                        l_.negated = True
                        if l_ in disj:
                            return self.mln.logic.true_false(1, mln=self.mln, index=self.index)
                        if l not in disj:
                            disj.append(l)
            if not conj:
                if len(disj) >= 2:
                    return self.mln.logic.disjunction(disj, mln=self.mln, index=self.index)
                else:
                    return disj[0].copy()
            if len(conj) == 1 and not disj:
                return conj[0].copy()
            conjuncts = conj[0].children
            remaining_disjuncts = disj + conj[1:]
            disj = []
            for c in conjuncts:
                disj.append(self.mln.logic.disjunction([c]+remaining_disjuncts, mln=self.mln, index=self.index))
            return self.mln.logic.conjunction(disj, mln=self.mln, index=self.index).cnf(level+1)

        def nnf(self, level=0):
            disjuncts = []
            for child in self.children:
                c = child.nnf(level+1)
                if isinstance(c, Logic.Disjunction):
                    disjuncts.extend(c.children)
                else:
                    disjuncts.append(c)
            return self.mln.logic.disjunction(disjuncts, mln=self.mln, index=self.index)

    class Literal(Formula):
        """
        represents a literal
        """
        def __init__(self, negated, pred_name, args, mln, index=None):
            # pdb.set_trace()
            Formula.__init__(self, mln, index)
            self.negated = negated
            self.pred_name = pred_name
            self.args = list(args)

        @property
        def negated(self):
            return self._negated

        @negated.setter
        def negated(self, value):
            self._negated = value

        @property
        def pred_name(self):
            return self._pred_name

        @pred_name.setter
        def pred_name(self, pred_name):
            self._pred_name = pred_name

        @property
        def args(self):
            return self._args

        @args.setter
        def args(self, args):
            self._args = args

        def __str__(self):
            return {True: '!', False: '', 2: '*'}[self.negated] + self.pred_name + "(" + ",".join(self.args) + ")"

        def var_doms(self, variables=None, constants=None):
            if variables is None:
                variables = {}
            arg_doms = self.mln.predicate(self.pred_name).arg_doms
            for i, arg in enumerate(self.args):
                if self.mln.logic.is_var(arg):
                    var_name = arg
                    domain = arg_doms[i]
                    variables[var_name] = domain
                elif constants is not None:
                    domain = arg_doms[i]
                    if domain not in constants:
                        constants[domain] = []
                    constants[domain].append(arg)
            return variables

        def template_variables(self, variables=None):
            if variables is None:
                variables = {}
            for i, arg in enumerate(self.args):
                if self.mln.logic.is_templ_var(arg):
                    var_name = arg
                    pred = self.mln.predicate(self.pred_name)
                    domain = pred.arg_doms[i]
                    variables[var_name] = domain
            return variables

        def pred_names(self, pred_names=None):
            if pred_names is None:
                pred_names = []
            if self.pred_name not in pred_names:
                pred_names.append(self.pred_name)
            return pred_names

        def ground(self, mrf, assignment, simplify=False, partial=False):
            args = [assignment.get(x, x) for x in self.args]
            if not any(map(self.mln.logic.is_var, args)):
                atom = "%s(%s)" % (self.pred_name, ",".join(args))
                ground_atom = mrf.ground_atom(atom)  # return
                if ground_atom is None:
                    raise Exception('Could not ground "%s". This atom is not among the ground atoms.' % atom)
                # simplify if necessary
                if simplify and ground_atom.truth(mrf.evidence) is not None:
                    truth = ground_atom.truth(mrf.evidence)
                    if self.negated:
                        truth = 1 - truth
                    return self.mln.logic.true_false(truth, mln=self.mln, index=self.index)
                gndformula = self.mln.logic.ground_literal(ground_atom, self.negated, mln=self.mln, index=self.index)
                return gndformula
            else:
                if partial:
                    return self.mln.logic.literal(self.negated, self.pred_name, args, mln=self.mln, index=self.index)
                if any([self.mln.logic.is_var(arg) for arg in args]):
                    raise Exception(
                        'Partial formula groundings are not allowed. Consider setting partial=True if desired.')
                else:
                    print("\nground atoms:")
                    mrf.print_ground_atoms()
                    raise Exception("Could not ground formula containing '%s' - this atom is not among the ground_atoms (see above)." % self.pred_name)

        def _ground_template(self, assignment):
            args = [assignment.get(x, x) for x in self.args]
            if self.negated == 2:  # template
                return [self.mln.logic.literal(False, self.pred_name, args, mln=self.mln),
                        self.mln.logic.literal(True, self.pred_name, args, mln=self.mln)]
            else:
                return [self.mln.logic.literal(self.negated, self.pred_name, args, mln=self.mln)]

        def constants(self, constants=None):
            if constants is None:
                constants = {}
            for i,c in enumerate(self.params):
                dom_name = self.mln.predicate(self.pred_name[0]).arg_doms[i]
                values = constants.get(dom_name, None)
                if values is None:
                    values = []
                    constants[dom_name] = values
                if not self.mln.logic.is_var(c) and not c in values: values.append(c)
            return constants

        def truth(self, world):
            return None

        def simplify(self, world):
            return self.mln.logic.literal(self.negated, self.pred_name, self.args, mln=self.mln, index=self.index)

        def copy(self, mln=None, index=inherit):
            return self.mln.logic.literal(self.negated, self.pred_name, self.args, mln=ifnone(mln, self.mln),
                                          index=self.index if index is inherit else index)

        def __eq__(self, other):
            return str(self) == str(other)

        def __ne__(self, other):
            return not self == other

    class LiteralGroup(Formula):
        """
        represents a group of literals with identical arguments
        """
        def __init__(self, negated, pred_name, args, mln, index=None):
            Logic.Formula.__init__(self, mln, index)
            self.negated = negated
            self.pred_name = pred_name
            self.args = args

        @property
        def negated(self):
            return self._negated

        @negated.setter
        def negated(self, value):
            self._negated = value

        @property
        def pred_name(self):
            return self._pred_name

        @pred_name.setter
        def pred_name(self, pred_name):
            self._pred_name = pred_name

        @property
        def literals(self):
            return [Literal(self.negated, literal, self.args, self.mln) for literal in self.pred_name]

        @property
        def args(self):
            return self._args

        @args.setter
        def args(self, args):
            self._args = args

        def var_doms(self, variables=None, constants=None):
            if variables is None:
                variables = {}
            arg_doms = self.mln.predicate(self.pred_name[0]).arg_doms
            for i, arg in enumerate(self.args):
                if self.mln.logic.is_var(arg):
                    var_name = arg
                    domain = arg_doms[i]
                    variables[var_name] = domain
                elif constants is not None:
                    domain = arg_doms[i]
                    if domain not in constants:
                        constants[domain] = []
                        constants[domain].append(arg)
            return variables

        def template_variables(self, variables=None):
            if variables is None:
                variables = {}
            for i, arg in enumerate(self.args):
                if self.mln.logic.is_templ_var(arg):
                    var_name = arg
                    pred = self.mln.predicate(self.pred_name[0])
                    domain = pred.arg_doms[i]
                    variables[var_name] = domain
            return variables

        def pred_names(self, pred_names=None):
            if pred_names is None:
                pred_names = []
            pred_names.extend([p for p in self.pred_name if p not in pred_names])
            return pred_names

        def _ground_template(self, assignment):
            if self.negated == 2:
                return [self.mln.logic.literal(False, pred_name, self.args, mln=self.mln) for pred_name in self.pred_name] + \
                       [self.mln.logic.literal(True, pred_name, self.args, mln=self.mln) for pred_name in self.pred_name]
            else:
                return [self.mln.logic.literal(self.negated, pred_name, self.args, mln=self.mln) for pred_name in
                        self.pred_name]

        def truth(self, world):
            return None

        def constants(self, constants=None):
            if constants is None:
                constants = {}
            for i, c in enumerate(self.params):
                dom_name = self.mln.predicate(self.pred_name[0]).arg_doms[i]
                values = constants.get(dom_name, None)
                if values is None:
                    values = []
                    constants[dom_name] = values
                if not self.mln.logic.is_var(c) and not c in values:
                    values.append(c)
            return constants

        def simplify(self, world):
            return self.mln.logic.literal_group(self.negated, self.pred_name, self.args, mln=self.mln,
                                                index=self.index)

        def __str__(self):
            return {True: '!', False: '', 2: '*'}[self.negated] + '|'.join(self.pred_name) + "(" + ",".join(
                self.args) + ")"

        def __eq__(self, other):
            return str(self) == str(other)

        def __ne__(self, other):
            return not self == other

    class GroundLiteral(Formula):
        """
        represents a ground literal
        """
        def __init__(self, ground_atom, negated, mln, index=None):
            Formula.__init__(self, mln, index)
            self.ground_atom = ground_atom
            self.negated = negated
            # self.mln = mln

        @property
        def ground_atom(self):
            return self._ground_atom

        @ground_atom.setter
        def ground_atom(self, ground_atom):
            self._ground_atom = ground_atom

        @property
        def negated(self):
            return self._negated

        @negated.setter
        def negated(self, negated):
            self._negated = negated

        @property
        def pred_name(self):
            return self.ground_atom.pred_name

        @property
        def args(self):
            return self.ground_atom.args

        def truth(self, world):
            tv = self.ground_atom.truth(world)
            if tv is None:
                return None
            if self.negated:
                return 1. - tv
            return tv

        def min_truth(self, world):
            truth = self.truth(world)
            if truth is None:
                return 0
            else:
                return truth

        def max_truth(self, world):
            truth = self.truth(world)
            if truth is None:
                return 1
            else:
                return truth

        def contains_ground_atom(self, atom_index):
            return self.ground_atom.index == atom_index

        def var_doms(self, variables=None, constants=None):
            return self.ground_atom.var_doms(variables, constants)

        def constants(self, constants=None):
            if constants is None:
                constants = {}
            for i,c in enumerate(self.ground_atom.args):
                dom_name = self.mln.predicates[self.ground_atom.pred_name][i]
                values = constants.get(dom_name, None)
                if values is None:
                    values = []
                    constants[dom_name] = values
                if not c in values: values.append(c)
            return constants

        def ground_atom_indices(self, l=None):
            if l is None:
                l = []
            if self.ground_atom.index not in l:
                l.append(self.ground_atom.index)
            return l

        def ground_atoms(self, l=None):
            if l is None:
                l = []
            if self.ground_atom not in l:
                l.append(self.ground_atom)
            return l

        def ground(self, mrf, assignment, simplify=False, partial=False):
            return self.mln.logic.ground_literal(mrf.ground_atom(str(self.ground_atom)), self.negated,
                                                 mln=self.mln, index=self.index)

        def copy(self, mln=None, index=inherit):
            mln = ifnone(mln, self.mln)
            if mln is not self.mln:
                # raise Exception('GroundLiteral cannot be copied among MLNs.')
                mln = self.mln
            return self.mln.logic.ground_literal(self.ground_atom, self.negated, mln=ifnone(mln, self.mln),
                                          index=self.index if index is inherit else index)

        def simplify(self, world):
            truth = self.truth(world)
            if truth is not None:
                return self.mln.logic.true_false(truth, mln=self.mln, index=self.index)
            return self.mln.logic.ground_literal(self.ground_atom, self.negated, mln=self.mln,
                                                 index=self.index)

        def pred_names(self, pred_names):
            if pred_names is None:
                pred_names = []
            if self.ground_atom.pred_name not in pred_names:
                pred_names.append(self.ground_atom.pred_name)
            return pred_names

        def template_variables(self, variables=None):
            return {}

        def _ground_template(self, assignment):
            return [self.mln.logic.ground_literal(self.ground_atom, self.negated, mln=self.mln)]

        def __str__(self):
            return {True: "!", False: ""}[self.negated] + str(self.ground_atom)

        def __eq__(self, other):
            return str(self) == str(other)  # self.negated == other.negated and self.ground_atom == other.ground_atom

        def __ne__(self, other):
            return not self == other

        def __hash__(self):  # hashable
            return hash(str(self))

    class GroundAtom:
        """
        represents a ground atom
        """

        def __init__(self, pred_name, args, mln, index=None):
            self.pred_name = pred_name
            self.args = args
            self.mln = mln
            self.index = index

        @property
        def pred_name(self):
            return self._pred_name

        @pred_name.setter
        def pred_name(self, pred_name):
            self._pred_name = pred_name

        @property
        def args(self):
            return self._args

        @args.setter
        def args(self, args):
            self._args = args

        @property
        def index(self):
            return self._index

        @index.setter
        def index(self, index):
            self._index = index

        def truth(self, world):
            return world[self.index]

        def min_truth(self, world):
            truth = self.truth(world)
            if truth is None:
                return 0
            else:
                return truth

        def max_truth(self, world):
            truth = self.truth(world)
            if truth is None:
                return 1
            else:
                return truth

        def pred_names(self, pred_names=None):
            if pred_names is None:
                pred_names = []
            if self.pred_name not in pred_names:
                pred_names.append(self.pred_name)
            return pred_names

        def var_doms(self, variables=None, constants=None):
            if variables is None:
                variables = {}
            if constants is None:
                constants = {}
            for d,c in zip(self.args, self.mln.predicate(self.pred_name).arg_doms):
                if d not in constants:
                    constants[d] = []
                if c not in constants:
                    constants[d].append(c)
            return variables

        def __repr__(self):
            return '<GroundAtom: %s>' % str(self)

        def __str__(self):
            return "%s(%s)" % (self.pred_name, ",".join(self.args))

        def __eq__(self, other):
            return str(self) == str(other)

        def __ne__(self, other):
            return not self == other

        def __hash__(self):  # hashable
            return hash(str(self))

    class Equality(ComplexFormula):
        """
        represents equality constraints
        """

        def __init__(self, args, negated, mln, index=None):
            Logic.ComplexFormula.__init__(self, mln, index)
            self.args = args
            self.negated = negated

        @property
        def args(self):
            return self._args

        @args.setter
        def args(self, args):
            self._args = args

        @property
        def negated(self):
            return self._negated

        @negated.setter
        def negated(self, negated):
            self._negated = negated

        def ground(self, mrf, assignment, simplify=False, partial=False):
            """
            if the parameter is a variable, do a lookup
            if a constant, use directly
            """
            args = map(lambda x: assignment.get(x, x), self.args)
            if self.mln.logic.is_var(args[0]) or self.mln.logic.is_var(args[1]):
                if partial:
                    return self.mln.logic.equality(args, self.negated, mln=self.mln)
            if simplify:
                equal = (args[0] == args[1])
                return self.mln.logic.true_false(1 if {True: not equal, False: equal}
                [self.negated] else 0, mln=self.mln, index=self.index)
            else:
                return self.mln.logic.equality(args, self.negated, mln=self.mln, index=self.index)

        def _ground_template(self, assignment):
            return [self.mln.logic.equality(self.args, negated=self.negated, mln=self.mln)]

        def template_variables(self, variables=None):
            return variables

        def var_doms(self, variables=None, constants=None):
            if variables is None:
                variables = {}
            if self.mln.logic.is_var(self.args[0]) and self.args[0] not in variables:
                variables[self.args[0]] = None
            if self.mln.logic.is_var(self.args[1]) and self.args[1] not in variables:
                variables[self.args[1]] = None
            return variables

        def var_dom(self, var_name):
            return None

        def var_domain_from_formula(self, formula):
            f_var_domains = formula.var_doms()
            eq_vars = self.var_doms()
            for var_ in eq_vars:
                eq_vars[var_] = f_var_domains
            return eq_vars

        def pred_names(self, pred_names=None):
            if pred_names is None:
                pred_names = []
            return pred_names

        def truth(self, world=None):
            if any(map(self.mln.logic.is_var, self.args)):
                return None
            equals = 1 if (self.args[0] == set.args[1]) else 0
            return (1 - equals) if self.negated else equals

        def max_truth(self, world):
            truth = self.truth(world)
            if truth is None:
                return 1
            else:
                return truth

        def min_truth(self, world):
            truth = self.truth(world)
            if truth is None:
                return 0
            else:
                return truth

        def simplify(self, world):
            truth = self.truth(world)
            if truth is not None:
                return self.mln.logic.true_false(truth, mln=self.mln, index=self.index)
            else:
                return self.mln.logic.equality(list(self.args), negated=self.negated, mln=self.mln,
                                               index=self.index)

    class Implication(ComplexFormula):
        """
        represents a implication
        """
        def __init__(self, children, mln, index=None):
            Formula.__init__(self, mln, index)
            self.children = children

        @property
        def children(self):
            return self._children

        @children.setter
        def children(self, children):
            self._children = children

        def __str__(self):
            c1 = self.children[0]
            c2 = self.children[1]
            return (str(c1) if not isinstance(c1, Logic.ComplexFormula) else '(%s)' % str(c1)) + " => " + \
                   (str(c2) if not isinstance(c2, Logic.ComplexFormula) else '(%s)' % str(c2))

        def cstr(self, color=False):
            c1 = self.children[0]
            c2 = self.children[1]
            (s1, s2) = (c1.cstr(color), c2.cstr(color))
            (s1, s2) = (('(%s)' if isinstance(c1, Logic.ComplexFormula) else '%s') % s1,
                        ('(%s)' if isinstance(c2, Logic.ComplexFormula) else '%s') % s2)
            return '%s => %s' % (s1, s2)

        def latex(self):
            return self.children[0].latex() + r" \rightarrow " + self.children[1].latex()

        def cnf(self, level=0):
            return self.mln.logic.disjunction(
                [self.mln.logic.negation([self.children[0]], mln=self.mln, index=self.index), self.children[1]],
                mln=self.mln, index=self.index).cnf(level + 1)

        def nnf(self, level=0):
            return self.mln.logic.disjunction(
                [self.mln.logic.negation([self.children[0]], mln=self.mln, index=self.index), self.children[1]],
                mln=self.mln, index=self.index).nnf(level + 1)

        def simplify(self, world):
            return self.mln.logic.disjunction(
                [Negation([self.children[0]], mln=self.mln, index=self.index), self.children[1]], mln=self.mln,
                index=self.index).simplify(world)

    class BiImplication(ComplexFormula):
        """
        represents a bi-implication
        """
        def __init__(self, children, mln, index=None):
            Formula.__init__(self, mln, index)
            self.children = children

        @property
        def children(self):
            return self._children

        @children.setter
        def children(self, children):
            self._children = children

        def __str__(self):
            c1 = self.children[0]
            c2 = self.children[1]
            return (str(c1) if not isinstance(c1, Logic.ComplexFormula) else '(%s)' % str(c1)) + " <=> " + \
                   (str(c2) if not isinstance(c2, Logic.ComplexFormula) else str(c2))

        def cnf(self, level=0):
            cnf = self.mln.logic.conjunction(
                [self.mln.logic.implication([self.children[0], self.children[1]], mln=self.mln, index=self.index),
                 self.mln.logic.implication([self.children[1], self.children[0]], mln=self.mln, index=self.index)],
                mln=self.mln, index=self.index)
            return cnf.cnf(level + 1)

        def nnf(self, level=0):
            return self.mln.logic.conjunction(
                [self.mln.logic.implication([self.children[0], self.children[1]], mln=self.mln, index=self.index),
                 self.mln.logic.implication([self.children[1], self.children[0]], mln=self.mln, index=self.index)],
                mln=self.mln, index=self.index).nnf(level + 1)

        def simplify(self, world):
            c1 = self.mln.logic.disjunction(
                [self.mln.logic.negation([self.children[0]], mln=self.mln), self.children[1]], mln=self.mln)
            c2 = self.mln.logic.disjunction(
                [self.children[0], self.mln.logic.negation([self.children[1]], mln=self.mln)], mln=self.mln)
            return self.mln.logic.conjunction([c1, c2], mln=self.mln, index=self.index).simplify(world)

    class Negation(ComplexFormula):
        """
        represents a negation of ComplexFormula
        """
        def __init__(self, children, mln, index=None):
            ComplexFormula.__init__(self, mln, index)
            if hasattr(children, '__iter__'):
                assert len(children) == 1
            else:
                children = children
            self.children = children

        @property
        def children(self):
            return self._children

        @children.setter
        def children(self, children):
            if hasattr(children, '__iter__'):
                if len(children) != 1:
                    raise Exception('Negation may have only 1 child.')
                else:
                    children = children
            self._children = children

        def truth(self, world):
            child_value = self.children[0].truth(world)
            if child_value is None:
                return None
            else:
                return 1 - child_value

        def cnf(self, level=0):
            """
            convert the formula that is negated to negation normal form (NNF),
            so that if it's a complex formula, it will be either a disjunction
            or conjunction, to which we can then apply De Morgan's law.
            Note: CNF conversion would be unnecessarily complex, and,
            when the children are negated below, most of it would be for nothing!
            """
            child = self.children[0].nnf(level + 1)
            if hasattr(child, "children"):
                neg_children = []
                for c in child.children:
                    neg_children.append(self.mln.logic.negation([c], mln=self.mln, index=None).cnf(level + 1))
                if isinstance(child, Logic.Conjunction):
                    return self.mln.logic.disjunction(neg_children, mln=self.mln, index=self.index).cnf(level + 1)
                elif isinstance(child, Logic.Disjunction):
                    return self.mln.logic.conjunction(neg_children, mln=self.mln, index=self.index).cnf(level + 1)
                elif isinstance(child, Logic.Negation):
                    return c.cnf(level + 1)
            elif isinstance(child, Logic.Literal):
                return self.mln.logic.literal(not child.negated, child.pred_name, child.args,
                                              mln=self.mln, index=self.index)
            elif isinstance(child, Logic.LiteralGroup):
                return self.mln.logic.literal_group(not child.negated, child.pred_name, child.args,
                                                    mln=self.mln, index=self.index)
            elif isinstance(child, Logic.GroundLiteral):
                return self.mln.logic.ground_literal(child.ground_atom, not child.negated, mln=self.mln,
                                                     index=self.index)
            elif isinstance(child, Logic.TrueFlase):
                return self.mln.logic.true_false(1-child.value, mln=self.mln, index=self.index)
            elif isinstance(child, Logic.Equality):
                return self.mln.logic.equality(child.params, not child.negated, mln=self.mln, index=self.index)

        def nnf(self, level=0):
            child = self.children[0].nnf(level + 1)
            if hasattr(child, 'children'):
                neg_children = []
                for c in child.children:
                    neg_children.append(self.mln.logic.negation([c], mln=self.mln, index=None).nnf(level + 1))
                if isinstance(child, Logic.Conjunction):         # !(A ^ B) = !A v !B
                    return self.mln.logic.disjunction(neg_children, mln=self.mln, index=self.index).nnf(level + 1)
                elif isinstance(child, Logic.Disjunction):       # !(A ^ B) = !A v !B
                    return self.mln.logic.conjunction(neg_children, mln=self.mln, index=self.index).nnf(level + 1)
                elif isinstance(child, Logic.Negation):
                    return c.nnf(level + 1)
            elif isinstance(child, Logic.Literal):
                return self.mln.logic.literal(not child.negated, child.pred_name, child.args,
                                              mln=self.mln, index=self.index)
            elif isinstance(child, Logic.LiteralGroup):
                return self.mln.logic.literal_group(not child.negated, child.pred_name, child.args,
                                                    mln=self.mln, index=self.index)
            elif isinstance(child, Logic.GroundLiteral):
                return self.mln.logic.ground_literal(child.ground_atom, not child.negated, mln=self.mln,
                                                     index=self.index)
            elif isinstance(child, Logic.TrueFalse):
                return self.mln.logic.true_false(1-child.value, mln=self.mln, index=self.index)
            elif isinstance(child, Logic.Equality):
                return self.mln.logic.equality(child.args, not child.negated, mln=self.mln,
                                               index=self.index)

        def simplify(self, world):
            f = self.children[0].simplify(world)
            if isinstance(f, Logic.TrueFalse):
                return f.invert()
            else:
                return self.mln.logic.negation([f], mln=self.mln, index=self.index)

        def __str__(self):
            return ('!(%s)' if isinstance(self.children[0], Logic.ComplexFormula) else '!%s') % str(self.children[0])

        def cstr(self, color=False):
            return ('!(%s)' if isinstance(self.children[0], Logic.ComplexFormula) else '!%s') % self.children[0].cstr(
                color)

    class Exist(ComplexFormula):
        def __init__(self, variables, formula, mln , index=None):
            Logic.Formula.__init__(self, mln, index)
            self.formula = formula
            self.vars = variables

        @property
        def children(self):
            return self._children

        @children.setter
        def children(self, children):
            self._children = children

        @property
        def formula(self):
            return self._children[0]

        @formula.setter
        def formula(self, f):
            self._children = [f]
        @property
        def vars(self):
            return self._vars

        @vars.setter
        def vars(self, v):
            self._vars = v

        def var_doms(self, variables=None, constants=None):
            if variables is None:
                variables = {}
            new_vars = self.formula.var_doms(None, constants)
            for var in self.vars:
                del new_vars[var]
            variables.update(dict([(k, v) for k, v in new_vars.items() if v is not None]))
            return variables

        def ground(self, mrf, assignment, partial=False, simplify=False):
            # find out variable domains
            var_doms = self.formula.var_doms()
            variables = dict([(k, v) for k, v in var_doms.items() if k in self.vars])
            groundings = []
            self._ground(self.children[0], variables, assignment, groundings, mrf, partial=partial)
            if len(groundings) == 1:
                return groundings[0]
            if not groundings:
                return self.mln.logic.true_false(0, mln=self.mln, index=self.index)
            disj = self.mln.logic.disjunction(groundings, mln=self.mln, index=self.index)
            if simplify:
                return disj.simplify(mrf.evidence)
            else:
                return disj

        def _ground(self, formula, variables, assignment, groundings, mrf, partial=False):
            # if all variables have been grounded
            if variables == {}:
                ground_formula = formula.ground(mrf, assignment, partial=partial)
                groundings.append(ground_formula)
                return
            var_name, dom_name = variables.popitem()
            for value in mrf.domains[dom_name]:
                assignment[var_name] = value
                self._ground(formula, dict(variables), assignment, groundings, mrf, partial=partial)

        def copy(self, mln=None, index=inherit):
            return self.mln.logic.exist(self.vars, self.formula, mln=ifnone(mln, self.mln),
                                        index=self.index if index is inherit else index)

        def truth(self, w):
            raise Exception("'%s' does not implement truth()" % self.__class__.__name__)

    class TrueFalse(Formula):
        """
        represent the constants truth values
        """
        def __init__(self, truth, mln, index=None):
            Formula.__init__(self, mln, index)
            self.value = truth

        @property
        def value(self):
            return self._value

        def cstr(self, color=False):
            return str(self)

        def truth(self, world=None):
            return self.value

        def min_truth(self, world=None):
            return self.truth

        def max_truth(self, world=None):
            return self.truth

        def invert(self):
            return self.mln.logic.true_false(1-self.truth(), mln=self.mln, index=self.index)

        def var_doms(self, variables=None, constants=None):
            if variables is None:
                variables = {}
            return variables

        def ground(self, mln, assignment, simplify=False, partial=False):
            return self.mln.logic.true_false(self.value, mln=self.mln, index=self.index)

        def copy(self, mln=None, index=inherit):
            return self.mln.logic.true_false(self.value, mln=ifnone(mln, self.mln),
                                             index=self.index if index is inherit else index)

        def simplify(self, world):
            return self.copy()

    class NonLogicalConstraint(Constraints):
        """
        A constraint that is not somehow made up of logical connectives and (ground) atoms.
        """
        def template_variants(self, mln):
            return [self]

        def is_logical(self):
            return False

        def negate(self):
            raise Exception("%s does not implement negate()" % str(type(self)))

    class CountConstraint(NonLogicalConstraint):
        """
        A constraint that tests the number of relation instances against an integer.
        """
        def __init__(self, predicate, predicate_params, fixed_params, op, count):
            """
            op : an operator, one of  "=",">=","<="
            """
            self.literal = self.mln.logic.literal(False, predicate, predicate_params)
            self.fixed_params = fixed_params
            self.count = count
            if op == "=":
                op = "=="
            self.op = op

        def __str__(self):
            op = self.op
            if op == "==": op = "="
            return "count(%s | %s) %s %d" % (str(self.literal), ", ".join(self.fixed_params),
                                             op, self.count)

        def cstr(self, color=False):
            return str(self)

        def iter_groundings(self, mrf, simplify=False):
            a = {}
            other_params = []
            for param in self.literal.params:
                if param[0].isupper():
                    a[param] = param
                else:
                    if param not in self.fixed_params:
                        other_params.append(param)
            for assignment in self._iter_assignment(mrf, list(self.fixed_params), a):
                ground_atoms = []
                for full_assignment in self._iter_assignment(mrf, list(other_params), assignment):
                    ground_literal = self.literal.ground(mrf, full_assignment, None)
                    ground_atoms.append(ground_literal.ground_atom)
                yield self.mln.logic.ground_count_constraint(ground_atoms, self.op, self.count), []

        def _iter_assignment(self, mrf, variables, assignment):
            """
            iterates over all possible assignments for the given variables of this constraint's literal
            variables: the variables that are still to be grounded
            """
            if len(variables) == 0:
                yield dict(assignment)
                return
            var_name = variables.pop()
            dom_name = self.literal.get_var_domain(var_name, mrf.mln)
            for value in mrf.domains[dom_name]:
                assignment[var_name] = value
                for a in self._iter_assignment(mrf, variables, assignment):
                    yield a

        def get_variables(self, mln, variables=None, constants=True):
            if constants is not None:
                self.literal.get_variables(mln, variables, constants)
            return variables

    class GroundCountConstraint(NonLogicalConstraint):
        def __init__(self, ground_atom, op, count):
            self.ground_atoms = ground_atom
            self.count = count
            self.op = op

        def is_true(self, world_values):
            c=0
            for ga in self.ground_atoms:
                if world_values[ga.index]:
                    c += 1
            return eval("c %s self.count" % self.op)

        def __str__(self):
            op = self.op
            if op == "==": op = "="
            return "count(%s) %s %d" % (";".join(map(str, self.ground_atoms)), op, self.count)

        def cstr(self, color=False):
            op = self.op
            if op == "==": op = "="
            return "count(%s) %s %d" % (";".join(map(lambda c: c.cstr(color), self.ground_atoms)), op, self.count)

        def negate(self):
            if self.op == "==":
                op = "!="
            elif self.op == "!=":
                op = "=="
            elif self.op == ">=":
                op = "<="
                self.count -= 1
            elif self.op == "<=":
                op = ">="
                self.count += 1

        def index_ground_atoms(self, l=None):
            if l is None: l = []
            for ga in self.ground_atoms:
                l.append(ga.index)
            return l

        def get_ground_atoms(self, l=None):
            if l is None: l = []
            for ga in self.ground_atoms:
                l.append(ga)
            return l

    def is_var(self, identifier):
        """
        Returns True if identifier is a logical variable according
        to the used grammar, and False otherwise.
        """
        return self.grammar.is_var(identifier)

    def is_constant(self, identifier):
        """
        Returns True if identifier is a logical constant according
        to the used grammar, and False otherwise.
        """
        return self.grammar.is_constant(identifier)

    def is_templ_var(self, s):
        return self.grammar.is_templ_var(s)

    def parse_formula(self, formula):
        # print "come to element's parse_formula"
        return self.grammar.parse_formula(formula)

    def parse_predicate(self, string):
        # pdb.set_trace()
        return self.grammar.parse_predicate(string)

    def parse_atom(self, string):
        return self.grammar.parse_atom(string)

    def parse_domain(self, dom):
        return self.grammar.parse_domain(dom)

    def parse_literal(self, l):
        return self.grammar.parse_literal(l)

    def is_literal(self, f):
        """
        Determines whether or not a formula is a literal.
        """
        return isinstance(f, Logic.GroundLiteral) or isinstance(f, Logic.Literal) or \
               isinstance(f, Logic.GroundAtom)

    def is_equal(self, f):
        """
        Determines whether or not a formula is an equality constraint.
        """
        return isinstance(f, Logic.Equality)

    def is_literal_conj(self, f):
        """
        Returns true if the given formula is a conjunction of literals.
        """
        if self.is_literal(f):
            return True
        if not isinstance(f, Logic.Conjunction):
            if not isinstance(f, Logic.Literal) and \
                    not isinstance(f, Logic.GroundLiteral) and \
                    not isinstance(f, Logic.Equality) and \
                    not isinstance(f, Logic.TrueFalse):
                return False
            return True
        for child in f.children:
            if not isinstance(child, Logic.Literal) and \
                    not isinstance(child, Logic.GroundLiteral) and \
                    not isinstance(child, Logic.Equality) and \
                    not isinstance(child, Logic.TrueFalse):
                return False
        return True

    def is_clause(self, f):
        """
        Returns true if the given formula is a clause (a disjunction of literals)
        """
        if self.is_literal(f):
            return True
        if not isinstance(f, Logic.Disjunction):
            if not isinstance(f, Logic.Literal) and \
                not isinstance(f, Logic.GroundLiteral) and \
                not isinstance(f, Logic.Equality) and \
                    not isinstance(f, Logic.TrueFalse):
                return False
            return True
        for child in f.children:
            if not isinstance(child, Logic.Literal) and \
                    not isinstance(child, Logic.GroundLiteral) and \
                    not isinstance(child, Logic.Equality) and \
                    not isinstance(child, Logic.TrueFalse):
                return False
        return True

    def negate(self, formula):
        """
        Returns a negation of the given formula.
        """
        if isinstance(formula, Logic.Literal) or isinstance(formula, Logic.GroundLiteral):
            ret = formula.copy()
            ret.negated = not ret.negated
        elif isinstance(formula, Logic.Equality):
            ret = formula.copy()
            ret.negated = not ret.negated
        else:
            ret = self.negation([formula.copy(mln=formula.mln, index=None)], mln=formula.mln,
                                index=formula.index)
        return ret

    @staticmethod
    def iter_eq_var_assignments(eq, f ,mln):
        """
        iterate over all variables assignments of an equality  constraint
        """
        doms = f.var_doms()
        eq_vars_ = eq.var_doms()
        if not set(eq_vars_).issubset(doms):
            raise Exception('Variable in (in)equality constraint not bound to a domain: %s' % eq)
        eq_vars = {}
        for v in eq_vars_:
            eq_vars[v] = doms[v]
        for assignment in Logic._iter_eq_var_assignments(mln, eq_vars, {}):
            yield  assignment

    @staticmethod
    def _iter_eq_var_assignments(mrf, variables, assignment):
        if len(variables) == 0:
            yield assignment
            return
        variables = dict(variables)
        variable, dom_name = variables.popitem()
        domain = mrf.domains[dom_name]
        for value in domain:
            for assignment in Logic._iter_eq_var_assignments(mrf, variables, dict_union(assignment, {variables: value})):
                yield assignment

    @staticmethod
    def clause_set(cnf):
        """
        Takes a formula in CNF and returns a set of clauses, i.e. a list of sets
        containing literals. All literals are converted into strings.
        """
        clauses = []
        if isinstance(cnf, Logic.Disjunction):
            clauses.append(set(map(str, cnf.children)))
        elif isinstance(cnf, Logic.Conjunction):
            for disj in cnf.children:
                clause = set()
                clauses.append(clause)
                if isinstance(disj, Logic.Disjunction):
                    for c in disj.children:
                        clause.add(str(c))
                else:
                    clause.add(str(disj))
        else:
            clauses.append(set([str(cnf)]))
        return clauses

    @staticmethod
    def cnf(gfs, formulas, logic, all_pos=False):
        """
        convert the given ground formulas to CNF
        if allPositive=True, then formulas with negative weights are negated to make all weights positive
        @return a new pair (ground_formulas, formulas)

        .. warning::

        If all_pos is True, this might have side effects on the formula weights of the MLN.
        """
        formulas_ = []
        negated = []
        if all_pos:
            for f in formulas:
                if f.weight < 0:
                    negated.append(f.index)
                    f = logic.negated(f)
                    f.weight = -f.weight
                formulas_.append(f)
        gfs_ = []
        for gf in gfs:
            if not gf.is_logical():
                if gf.index is negated:
                    gf.negate()
                gfs_.append(gf)
                continue
            if gf.index in negated:
                cnf = logic.negate(gf).cnf()
            else:
                cnf = gf.cnf()
            if isinstance(cnf, Logic.TrueFalse):
                continue
            cnf.index = gf.index
            gfs_.append(cnf)
        return gfs_, formulas_

    def conjugate(self, children, mln=None, index=inherit):
        """
        Returns a conjunction of the given children.
        Performs rudimentary simplification in the sense that if children
        has only one element, it returns this element (e.g. one literal)
        """
        if not children:
            return self.true_false(0, mln=ifnone(mln, self.mln), index=index)
        elif len(children) == 1:
            return children[0].copy(mln=ifnone(mln, self.mln), index=index)
        else:
            return self.conjunction(children, mln=ifnone(mln, self.mln), index=index)

    def disjugate(self, children, mln=None, index=inherit):
        """
        Returns a conjunction of the given children.
        Performs rudimentary simplification in the sense that if children
        has only one element, it returns this element (e.g. one literal)
        """
        if not children:
            return self.true_false(0, mln=ifnone(mln, self.mln), index=index)
        elif len(children) == 1:
            return children[0].copy(mln=ifnone(mln, self.mln), index=index)
        else:
            return self.disjunction(children, mln=ifnone(mln, self.mln), index=index)

    def conjunction(self, *args, **kwargs):
        """
        Returns a new instance of a Conjunction object.
        """
        raise Exception('%s does not implement conjunction()' % str(type(self)))

    def disjunction(self, *args, **kwargs):
        """
        Returns a new instance of a Disjunction object.
        """
        raise Exception('%s does not implement disjunction()' % str(type(self)))

    def negation(self, *args, **kwargs):
        """
        Returns a new instance of a Negation object.
        """
        raise Exception('%s does not implement negation()' % str(type(self)))

    def implication(self, *args, **kwargs):
        """
        Returns a new instance of a Implication object.
        """
        raise Exception('%s does not implement implication()' % str(type(self)))

    def bi_implication(self, *args, **kwargs):
        """
        Returns a new instance of a Bi_implication object.
        """
        raise Exception('%s does not implement bi_implication()' % str(type(self)))

    def equality(self, *args, **kwargs):
        """
        Returns a new instance of a Equality object.
        """
        raise Exception('%s does not implement equality()' % str(type(self)))

    def exist(self, *args, **kwargs):
        """
        Returns a new instance of a Exist object.
        """
        raise Exception('%s does not implement exist()' % str(type(self)))

    def ground_atom(self, *args, **kwargs):
        """
        Returns a new instance of a ground_atom object.
        """
        raise Exception('%s does not implement ground_atom()' % str(type(self)))

    def literal(self, *args, **kwargs):
        """
        Returns a new instance of a Lit object.
        """
        raise Exception('%s does not implement literal()' % str(type(self)))

    def literal_group(self, *args, **kwargs):
        """
        Returns a new instance of a Lit object.
        """
        raise Exception('%s does not implement literal_group()' % str(type(self)))

    def ground_literal(self, *args, **kwargs):
        """
        Returns a new instance of a GndLit object.
        """
        raise Exception('%s does not implement ground_literal()' % str(type(self)))

    def count_constraint(self, *args, **kwargs):
        """
        Returns a new instance of a CountConstraint object.
        """
        raise Exception('%s does not implement count_constraint()' % str(type(self)))

    def true_false(self, *args, **kwargs):
        """
        Returns a new instance of a TrueFalse constant object.
        """
        raise Exception('%s does not implement true_false()' % str(type(self)))

    def create(self, clazz, *args, **kwargs):
        """
        Takes the type of a logical element (class type) and creates
        a new instance of it.
        """
        return clazz(*args, **kwargs)


# this is a little hack to make nested classes pickle-able
Constraints = Logic.Constraints
Formula = Logic.Formula
ComplexFormula = Logic.ComplexFormula
Conjunction = Logic.Conjunction
Disjunction = Logic.Disjunction
Literal = Logic.Literal
LiteralGroup = Logic.LiteralGroup
GroundLiteral = Logic.GroundLiteral
GroundAtom = Logic.GroundAtom
Equality = Logic.Equality
Implication = Logic.Implication
BiImplication = Logic.BiImplication
Negation = Logic.Negation
Exist = Logic.Exist
TrueFalse = Logic.TrueFalse
NonLogicalConstraint = Logic.NonLogicalConstraint
CountConstraint = Logic.CountConstraint
GroundCountConstraint = Logic.GroundCountConstraint


