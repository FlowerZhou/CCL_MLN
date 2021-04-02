"""
Markov random field
"""
import itertools
from collections import defaultdict
from dnutils import logs, out

from mln.database import DataBase
from mln.util import merge_dom, CallbyRef, get_index
import copy
import sys
import re
from mln.util import fstr
from logic.fol import FirstOrderLogic
from mln.util import logx
import time
from mln.grounding import *
from logic.elements import Logic
from mln.constants import HARD, nan
from mln.mrfvars import MutexVariable, SoftMutexVariable, FuzzyVariable, \
    BinaryVariable
from mln.util import CallbyRef, Interval, temporary_evidence, tty
from mln.method import InferenceMethods
from math import *
import traceback
import pdb
from mln.tail_call_optimized import tail_call_optimized

logger = logs.getlogger(__name__)
sys.setrecursionlimit(20000)


class MRF(object):
    """
    Represents a ground Markov random field.
    :_ground_atoms:         dict mapping a string representation of a ground atom to its Logic.GroundAtom object
    :_ground_atoms_indices: dict mapping ground atom index to Logic.GroundAtom object
    :_evidence:             vector of evidence truth values of all ground atoms
    :_variables:            dict mapping variable names to their class instance.

    :param mln:    the MLN tied to this MRF.
    :param db:     the database that the MRF shall be grounded with.
    """

    def __init__(self, mln, db):
        if not mln._materialized:
            self.mln = mln.materialize(db)
        else:
            self.mln = mln
        self._evidence = []
        self._variables = {}
        self._variables_by_index = {}  # ground atom index -> variable
        self._variables_by_ground_atom_index = {}  # ground atom index
        self._ground_atoms = {}
        self._ground_atoms_by_index = {}
        # get combined domain
        self.domains = merge_dom(self.mln.domains, db.domains)
        # soft evidence and can be handled in exactly the same way
        # ground members
        self.formulas = list(self.mln.formulas)
        if isinstance(db, str):
            db = DataBase.load(self.mln, dbfile=db)
        elif isinstance(db, DataBase):
            pass
        elif db is None:
            db = DataBase(self.mln)
        else:
            raise Exception("Not a valid database argument (type %s)" % (str(type(db))))
        self.db = db
        # materialize formula weights
        self._materialize_weights()
        return

    @property
    def variables(self):
        return sorted(list(self._variables.values()), key=lambda v: v.index)

    @property
    def ground_atoms(self):
        return list(self._ground_atoms.values())

    @property
    def evidence(self):
        return self._evidence

    @evidence.setter
    def evidence(self, evidence):
        self._evidence = evidence
        self.consistent()

    @property
    def predicates(self):
        return self.mln.predicates

    @property
    def hard_formulas(self):
        """
        Returns a list of all hard formulas in this MRF.
        """
        return [f for f in self.formulas if f.weight == HARD]

    def _get_pred_groundings(self, pred_name):
        """
        Gets the names of all ground atoms of the given predicate.
        """
        # get the string representation of the first grounding of the predicate
        if pred_name not in self.predicates:
            raise Exception('Unknown predicate "%s" (%s)' % (pred_name, map(str, self.predicates)))
        dom_names = self.predicates[pred_name]
        params = []
        for domName in dom_names:
            params.append(self.domains[domName][0])
        ground_atom = "%s(%s)" % (pred_name, ",".join(params))
        # get all subsequent groundings (by index) until the predicate name changes
        groundings = []
        index = self.ground_atoms[ground_atom].index
        while True:
            groundings.append(ground_atom)
            index += 1
            if index >= len(self.ground_atoms):
                break
            ground_atom = str(self.ground_atoms_by_index[index])
            if self.mln.logic.parse_atom(ground_atom)[0] != pred_name:
                break
        return groundings

    def _get_pred_groundings_as_indices(self, pred_name):
        """
        Get a list of all the indices of all groundings of the given predicate
        """
        # get the index of the first grounding of the predicate and the number of groundings
        dom_names = self.predicates[pred_name]
        params = []
        num_groundings = 1
        for domName in dom_names:
            params.append(self.domains[domName][0])
            num_groundings *= len(self.domains[domName])
        ground_atom = "%s(%s)" % (pred_name, ",".join(params))
        if ground_atom not in self.ground_atoms:
            return []
        index_first = self.ground_atoms[ground_atom].index
        return list(range(index_first, index_first + num_groundings))

    def dom_size(self, dom_name):
        if dom_name not in self.domains:
            raise Exception("No such domname %s" %dom_name)
        return len(self.domains[dom_name])

    def _materialize_weights(self, verbose=False):
        """
        materialize all formula weights.
        """
        max_weight = 0
        for f in self.formulas:
            if f.weight is not None and f.weight != HARD:
                w = str(f.weight)
                variables = re.findall(r'\$\w+', w)
                for var in variables:
                    try:
                        w, numReplacements = re.subn(r'\%s' % var, self.mln.vars[var], w)
                    except Exception:
                        raise Exception("Error substituting variable references in '%s'\n" % w)
                    if numReplacements == 0:
                        raise Exception("Undefined variable(s) referenced in '%s'" % w)
                w = re.sub(r'domSize\((.*?)\)', r'self.domsize("\1")', w)
                try:
                    f.weight = float(eval(w))
                except:
                    sys.stderr.write("Evaluation error while trying to compute '%s'\n" % w)
                    raise
                max_weight = max(abs(f.weight), max_weight)

    def __getitem__(self, key):
        return self.evidence[self.ground_atom(key).index]

    def __setitem__(self, key, value):
        self.set_evidence({key: value}, erase=False)

    def set_evidence(self, atom_values, erase=False, cw=False):
        """
        Sets the evidence of variables in this MRF.

        If erase is `True`, for every ground atom appearing in atom_values, the truth values of all ground
        ground atom in the respective MRF variable are erased before the evidences are set. All other ground
        atoms stay untouched.
        :param atom_values:     a dict mapping ground atom strings/objects/indices to their truth
                               values.
        :param erase:          specifies whether or not variables shall be erased before asserting the evidences.
                               Only affects the variables that are present in `atom_values`.
        :param cw:             applies the closed-world assumption for all non evidence atoms.
        """
        # check validity of evidence values
        atom_values_ = {}
        for key, value in dict(atom_values).items():
            # convert boolean to numeric values
            if value in (True, False):
                atom_values[key] = {True: 1, False: 0}[value]
                value = atom_values[key]
            ground_atom = self.ground_atom(key)
            atom_values_[str(ground_atom)] = value
            var = self.variable(ground_atom)
        atom_values = atom_values_
        if erase:  # erase all variable assignments appearing in atom_values
            for key, _ in atom_values.items():
                var = self.variable(self.ground_atom(key))
                # unset all atoms in this variable
                for atom in var.ground_atoms:
                    self._evidence[atom.index] = None

        for key, value in atom_values.items():
            ground_atom = self.ground_atom(key)
            var = self.variable(ground_atom)
            # create a template with admissible truth values for all
            # ground atoms in this variable
            values = [-1] * len(var.ground_atoms)
            for _, val in var.iter_values(evidence={ground_atom.index: value}):
                for i, (v, v_) in enumerate(zip(values, val)):
                    if v == -1:
                        values[i] = v_
                    elif v is not None and v != v_:
                        values[i] = None
            for atom, val in zip(var.ground_atoms, values):
                curval = self._evidence[atom.index]
                if curval is not None and val is not None and curval != val:
                    raise Exception(
                        'Contradictory evidence in variable %s: %s = %s vs. %s' % (var.name, str(ground_atom), curval, val))
                elif curval is None and val is not None:
                    self._evidence[atom.index] = val
        # pdb.set_trace()
        if cw:
            self.apply_cw()

    def erase(self):
        """
        Erases all evidence in the MRF.
        """
        self._evidence = [None] * len(self.ground_atoms)

    def apply_cw(self, *pred_names):
        """
        Applies the closed world assumption to this MRF.

        Sets all evidences to 0 if they don't have truth value yet.

        :param pred_names:     a list of predicate names the cw assumption shall be applied to.
                              If empty, it is applied to all predicates.
        """
        for i, v in enumerate(self._evidence):
            if pred_names and self.ground_atom(i).pred_name not in pred_names:
                continue
            if v is None:
                self._evidence[i] = 0

    def consistent(self, strict=False):
        """
        Performs a consistency check on this MRF wrt. to the variable value assignments.
        Raises an MRFValueException if the MRF is inconsistent.
        """
        for variable in self.variables:
            variable.consistent(self.evidence_dicti(), strict=strict)

    def ground_atom(self, identifier, *args):
        """
        Returns the the ground atom instance that is associated with the given identifier, or adds a new ground atom.
        :param identifier:    Either the string representation of the ground atom or its index (int)
        :returns:             the :class:`logic.common.Logic.GroundAtom` instance or None, if the ground
                              atom doesn't exist.

        :Example:
        >>> mrf = MRF(mln)
        >>> mrf.ground_atom('foo', 'x', 'y') / mrf.ground_atom('foo(x,y)') / mrf.ground_atom(0)
        # add the ground atom 'foo(x,y)'
        """
        if not args:
            if isinstance(identifier, str):
                atom = self._ground_atoms.get(identifier)
                if atom is None:
                    try:
                        _, pred_name, args = self.mln.logic.parse_literal(identifier)
                    except Exception:
                        return None
                    atom_str = str(self.mln.logic.ground_atom(pred_name, args, self.mln))
                    return self._ground_atoms.get(atom_str)
                else:
                    return atom
            elif type(identifier) is int:
                return self._ground_atoms_by_index.get(identifier)
            elif isinstance(identifier, Logic.GroundAtom):
                return self._ground_atoms.get(str(identifier))

            else:
                raise Exception('Illegal identifier type: %s' % type(identifier))
        else:
            return self.new_ground_atom(identifier, *args)

    def variable(self, identifier):
        """
        Returns the class instance of the variable with the name or index `var`,
        or None, if no such variable exists.

        :param identifier:    (string/int/:class:`logic.common.Logic.GroundAtom`) the name or index of the variable,
                              or the instance of a ground atom that is part of the desired variable.
        """
        if type(identifier) is int:
            return self._variables_by_index.get(identifier)
        elif isinstance(identifier, Logic.GroundAtom):
            return self._variables_by_ground_atom_index[identifier.index]
        elif isinstance(identifier, str):
            return self._variables.get(identifier)

    def new_ground_atom(self, pred_name, *args):
        """
        Adds a ground atom to the set (actually it's a dict) of ground atoms.

        If the ground atom is already in the MRF it does nothing but returning the existing
        ground atom instance. Also updates/adds the variables of the MRF.

        :param pred_name:    the predicate name of the ground atom
        :param *args:       the list of predicate arguments `logic.common.Logic.GroundAtom` object
        """
        # create and add the ground atom
        ground_atom = self.mln.logic.ground_atom(pred_name, args, self.mln)
        if str(ground_atom) in self._ground_atoms:
            return self._ground_atoms[str(ground_atom)]
        self._evidence.append(None)
        ground_atom.index = len(self._ground_atoms)
        self._ground_atoms[str(ground_atom)] = ground_atom
        self._ground_atoms_by_index[ground_atom.index] = ground_atom
        # add the ground atom to the variable it belongs
        # to or create a new one if it doesn't exists.
        predicate = self.mln.predicate(ground_atom.pred_name)
        var_name = predicate.var_name(ground_atom)
        variable = self.variable(var_name)
        if variable is None:
            variable = predicate.to_variable(self, var_name)
            self._variables[variable.name] = variable
            self._variables_by_index[variable.index] = variable
        variable.ground_atoms.append(ground_atom)
        self._variables_by_ground_atom_index[ground_atom.index] = variable
        return ground_atom

    def print_variables(self):
        for var in self.variables:
            print(str(var))

    def print_world_atoms(self, world, stream=sys.stdout):
        """
        Prints the given world `world` as a readable string of the plain gnd atoms to the given stream.
        """
        for ground_atom in self.ground_atoms:
            v = world[ground_atom.index]
            vstr = '%.3f' % v if v is not None else '?    '
            stream.write('%s  %s\n' % (vstr, str(ground_atom)))

    def print_world_vars(self, world, stream=sys.stdout, tb=2):
        """
        Prints the given world `world` as a readable string of the MRF variables to the given stream.
        """
        out('=== WORLD VARIABLES ===', tb=tb)
        for var in self.variables:
            stream.write(repr(var) + '\n')
            for i, v in enumerate(var.evidence_value(world)):
                vstr = '%.3f' % v if v is not None else '?    '
                stream.write('  %s  %s\n' % (vstr, var.ground_atoms[i]))

    def print_domains(self):
        out('=== MRF DOMAINS ==', tb=2)
        for dom, values in self.domains.items():
            print(dom, '=', ','.join(values))

    def evidence_dicts(self):
        """
        Returns, from the current evidence list, a dictionary that maps ground atom names to truth values
        """
        d = {}
        for index, tv in enumerate(self._evidence):
            d[str(self._ground_atoms_by_index[index])] = tv
        return d

    def evidence_dicti(self):
        """
        Returns, from the current evidence list, a dictionary that maps ground atom indices to truth values
        """
        d = {}
        for index, tv in enumerate(self._evidence):
            d[index] = tv
        return d

    def count_worlds(self, with_evidence=True):
        """
        Computes the number of possible worlds this MRF can take.

        :param with_evidence:    (bool) if True, takes into account the evidence which is currently set in the MRF.
                                if False, computes the total number of possible worlds.

        .. note:: this method does not enumerate the possible worlds.
        """
        worlds = 1
        ev = self.evidence_dicti if with_evidence else {}
        for var in self.variables:
            worlds *= var.value_count(ev)
        return worlds

    def iter_worlds(self):
        """
        Iterates over the possible worlds of this MRF taking into account the evidence vector of truth values.

        :returns: a generator of (index, possible world) tuples.
        """
        for res in self._iter_worlds([v for v in self.variables if v.value_count(self.evidence) > 1], list(self.evidence),
                                    CallbyRef(0), self.evidence_dicti()):
            yield res

    @tail_call_optimized
    def _iter_worlds(self, variables, world, world_index, evidence):
        if not variables:
            yield world_index.value, world
            world_index.value += 1
            return
        # pdb.set_trace()
        variable = variables[0]
        if isinstance(variable, FuzzyVariable):
            world_ = list(world)
            value = variable.evidence_value(evidence)
            for res in self._iter_worlds(variables[1:], variable.setval(value, world_), world_index, evidence):
                yield res
        else:
            for _, value in variable.iter_values(evidence):
                world_ = list(world)
                for res in self._iter_worlds(variables[1:], variable.setval(value, world_), world_index, evidence):
                    yield res

    def _iter_worlds_test(self, variables, world, world_index, evidence):
        pass

    def worlds(self):
        """
        Iterates over all possible worlds (taking evidence into account).

        :returns:    a generator of possible worlds.
        """
        for _, world in self.iter_worlds():
            yield world

    def iter_all_worlds(self):
        """
        Iterates over all possible worlds (without) taking evidence into account).

        :returns:    a generator of possible worlds.
        """
        world = [None] * len(self.evidence)
        for i, w in self._iter_worlds(self.variables, world, CallbyRef(0), {}):
            yield i, w

    def print_evidence_atoms(self, stream=sys.stdout):
        """
        Prints the evidence truth values of plain ground atoms to the given `stream`.
        """
        self.print_world_atoms(self.evidence, stream)

    def print_evidence_vars(self, stream=sys.stdout):
        """
        Prints the evidence truth values of the variables of this MRF to the given `stream`.
        """
        self.print_world_vars(self.evidence, stream, tb=3)

    def print_ground_atoms(self, stream=sys.stdout):
        """
        Prints the alphabetically sorted list of ground atoms in this MRF to the given `stream`.
        """
        out('=== GROUND ATOMS ===', tb=2)
        l = list(self._ground_atoms.keys())
        for ga in sorted(l):
            stream.write(str(ga) + '\n')

    def _weights(self):
        # returns the weight vector as a list
        return [f.weight for f in self.formulas]

    def iter_groundings(self, simplify=False, grounding_factory='DefaultGroundingFactory'):
        """
        Iterates over all groundings of all formulas of this MRF.

        :param simplify:  if True, the ground formulas are simplified wrt to the evidence in the MRF.
        :param grounding_factory: the grounding factory to be used.
        """
        grounder = eval('%s(self, simplify=simplify)' % grounding_factory)
        for ground_formula in grounder.itergroundings():
            yield ground_formula

    def reduce_variables(self, variables, query):
        """
        construct a reduced mrf, only construct a minimal subset of the ground network
        """
        # pdb.set_trace()
        re_variables = []
        query_ = []
        for i, _ in enumerate(query):
            query_.append(query[i].ground_atom)
        query_ = set(query_)
        mrf = query_
        evidence_variables = set()
        for i, v in enumerate(self.evidence):
            if self._evidence[i] is not None:
                evidence_variables.add(self._variables_by_index[i].ground_atoms[0])
        # pdb.set_trace()
        while len(query_):
            for q in query_.copy():
                if q not in evidence_variables:
                    mrb = self.get_markov_blanket(q)
                    # mrb.difference_update(mrf)
                    query_ = query_.union(mrb)
                    query_.difference_update(mrf)
                    mrf = mrf.union(mrb)
                query_.discard(q)
        # pdb.set_trace()
        for atom in mrf:
            for var in variables:
                if atom == var.ground_atoms[0]:
                    re_variables.append(var)
        # pdb.set_trace()
        reduce_rate = (len(variables)-len(re_variables))/float(len(variables))
        print("%.2f%% nodes can be reduced !" % (reduce_rate * 100))

        return re_variables

    def get_markov_blanket(self, variable):
        """
        get a markov blanket of a node
        """
        # pdb.set_trace()
        mrb = set()
        base = None
        children = []
        for f in self.formulas:
            predicate = variable.pred_name
            if not hasattr(f, 'children'):
                continue
            for child in f.children:
                if child.pred_name == predicate:
                    base = child
                    break
            if base is not None:
                children = f.children[:]
                children.remove(base)
                for child in children:
                    if len(child.args) >= len(base.args):
                        index = get_index(child.args, base.args)
                        for ground_atom in self.ground_atoms:
                            flag = True
                            if ground_atom.pred_name == child.pred_name:
                                for i, idx in enumerate(index):
                                    if ground_atom.args[idx] != variable.args[i]:
                                        flag = False
                                        break
                                if flag:
                                    if ground_atom != variable:
                                        mrb.add(ground_atom)
                    else:
                        index = get_index(base.args, child.args)
                        for ground_atom in self.ground_atoms:
                            flag = True
                            if ground_atom.pred_name == child.pred_name:
                                for i, idx in enumerate(index):
                                    if ground_atom.args[i] != variable.args[idx]:
                                        flag = False
                                        break
                                if flag:
                                    mrb.add(ground_atom)
                base = None
        return mrb









