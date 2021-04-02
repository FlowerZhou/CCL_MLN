import math
import random
from collections import defaultdict

from dnutils import logs, ProgressBar, out

from .mcmc import MCMCInference, active_variables
from ..constants import ALL, HARD
from ..grounding.fastconj import FastConjunctionGrounding
from mln.util import item
from logic.elements import Logic
import pdb
from numba import jit

logger = logs.getlogger(__name__)
active_clauses = []  # record active clauses' index


class MCSAT(MCMCInference):
    """ 
    MC-SAT/MC-SAT-PC
    """

    def __init__(self, mrf, queries=ALL, **params):
        MCMCInference.__init__(self, mrf, queries, **params)
        self._weight_backup = list(self.mrf.mln.weights)

    def _initkb(self, verbose=False):
        """
        Initialize the knowledge base to the required format and collect structural information for optimization purposes
        """
        # convert the MLN ground formulas to CNF
        logger.debug("converting formulas to cnf...")
        # self.mln._toCNF(allPositive=True)
        self.formulas = []
        for f in self.mrf.formulas:
            if f.weight < 0:
                f.weight = -f.weight
                f = self.mln.logic.negate(f)
            self.formulas.append(f)
      
        grounder = FastConjunctionGrounding(self.mrf, formulas=self.formulas, simplify=True, verbose=self.verbose)
        self.ground_formulas = []
        for gf in grounder.iter_groundings():
            if isinstance(gf, Logic.TrueFalse):
                continue
            # pdb.set_trace()
            self.ground_formulas.append(gf.cnf())
        self._watch.tags.update(grounder.watch.tags)
        # pdb.set_trace()
        # get clause data
        logger.debug("gathering clause data...")
        self.gf2clauseindex = {}  # ground formula index -> tuple (indexFirstClause, indexLastClause+1) for use with range
        self.clauses = []  # list of clauses, where each entry is a list of ground literals
        # self.GAoccurrences = {} # ground atom index -> list of clause indices (into self.clauses)
        i_clause = 0
        # process all ground formulas
        for i_gf, gf in enumerate(self.ground_formulas):
            # get the list of clauses
            if isinstance(gf, Logic.Conjunction):
                clauses = [clause for clause in gf.children if not isinstance(clause, Logic.TrueFalse)]
            elif not isinstance(gf, Logic.TrueFalse):
                clauses = [gf]
            else:
                continue
            self.gf2clauseindex[i_gf] = (i_clause, i_clause + len(clauses))
            # process each clause
            for c in clauses:
                if hasattr(c, "children"):
                    literals = c.children
                    # get the active clauses
                    """
                    for index in active_variables:
                        if self.mrf.variables[index].ground_atoms in [lt.ground_atom for lt in literals]:
                            active_clauses.append(i_gf)
                    """
                else:  # unit clause
                    literals = [c]
                    """
                    if c.ground_atom == self.mrf.variables[index].ground_atoms:
                        active_clauses.append(i_gf)
                    """
                # add clause to list
                self.clauses.append(literals)
                # next clause index
                i_clause += 1
        # add clauses for soft evidence atoms
        for se in []:  # self.softEvidence:
            se["numTrue"] = 0.0
            formula = self.mln.logic.parseFormula(se["expr"])
            se["formula"] = formula.ground(self.mrf, {})
            cnf = formula.toCNF().ground(self.mrf, {})
            indexFirst = i_clause
            for clause in self._formula_clauses(cnf):
                self.clauses.append(clause)
                # print clause
                i_clause += 1
            se["indexClausePositive"] = (indexFirst, i_clause)
            cnf = self.mln.logic.negation([formula]).toCNF().ground(self.mrf, {})
            indexFirst = i_clause
            for clause in self._formula_clauses(cnf):
                self.clauses.append(clause)
                # print clause
                i_clause += 1
            se["indexClauseNegative"] = (indexFirst, i_clause)

    def _formula_clauses(self, f):
        # get the list of clauses
        if isinstance(f, Logic.Conjunction):
            lc = f.children
        else:
            lc = [f]
        # process each clause
        for c in lc:
            if hasattr(c, "children"):
                yield c.children
            else:  # unit clause
                yield [c]

    @property
    def chains(self):
        return self._params.get('chains', 1)

    @property
    def maxsteps(self):
        return self._params.get('maxsteps', 100)

    @property
    def softevidence(self):
        return self._params.get('softevidence', False)

    @property
    def use_se(self):
        return self._params.get('use_se')

    @property
    def p(self):
        return self._params.get('p', .5)

    @property
    def resulthistory(self):
        return self._params.get('resulthistory', False)

    @property
    def historyfile(self):
        return self._params.get('historyfile', None)

    @property
    def rndseed(self):
        return self._params.get('rndseed', None)

    @property
    def initalgo(self):
        return self._params.get('initalgo', 'SampleSAT')

    def _run(self):
        """
        p: probability of a greedy (WalkSAT) move
        initAlgo: algorithm to use in order to find an initial state that satisfies all hard constraints ("SampleSAT"
        or "SAMaxWalkSat")
        verbose: whether to display results upon completion
        details: whether to display information while the algorithm is running            
        infoInterval: [if details==True] interval (no. of steps) in which to display the current step number and some
        additional info
        resultsInterval: [if details==True] interval (no. of steps) in which to display intermediate results;
        [if keepResultsHistory==True] interval in which to store intermediate results in the history
        debug: whether to display debug information (e.g. internal data structures) while the algorithm is running
            debugLevel: controls degree to which debug information is presented
        keepResultsHistory: whether to store the history of results (at each resultsInterval)
        referenceResults: reference results to compare obtained results to
        saveHistoryFile: if not None, save history to given filename
        sampleCallback: function that is called for every sample with the sample and step number as parameters
        softEvidence: if None, use soft evidence from MLN, otherwise use given dictionary of soft evidence
        handleSoftEvidence: if False, ignore all soft evidence in the MCMC sampling (but still compute softe evidence
        statistics if soft evidence is there)
        """
        logger.debug("starting MC-SAT with maxsteps=%d, softevidence=%s" % (self.maxsteps, self.softevidence))
        # initialize the KB and gather required info
        # pdb.set_trace()
        self._initkb()
        # print CNF KB
        logger.debug("CNF KB:")
        for gf in self.ground_formulas:
            logger.debug("%7.3f  %s" % (gf.weight, str(gf)))
        print()
        # pdb.set_trace()
        # set the random seed if it was given
        if self.rndseed is not None:
            random.seed(self.rndseed)
        # create chains
        chaingroup = MCMCInference.ChainGroup(self)

        self.chaingroup = chaingroup
        for i in range(self.chains):
            chain = MCMCInference.Chain(self, self.queries)
            chaingroup.chain(chain)
            # satisfy hard constraints using initialization algorithm
            M = []  # clause_indices
            NLC = []   # nlcs
            for i, gf in enumerate(self.ground_formulas):
                if gf.weight == HARD:
                    if gf.islogical():
                        clause_range = self.gf2clauseindex[i]
                        M.extend(list(range(*clause_range)))
                    else:
                        NLC.append(gf)
            if M or NLC:
                logger.debug('Running SampleSAT')
                chain.state = SampleSAT(self.mrf, chain.state, M, NLC, self,
                                        p=self.p).run()
                # Note: can't use p=1.0 because there is a chance of getting into an oscillating state
        if logger.level == logs.DEBUG:
            self.mrf.print_world_vars(chain.state)
        self.step = 1
        logger.debug('running MC-SAT with %d chains' % len(chaingroup.chains))
        self._watch.tag('running MC-SAT', self.verbose)
        if self.verbose:
            bar = ProgressBar(steps=self.maxsteps, color='green')
        while self.step <= self.maxsteps:
            # take one step in each chain
            for chain in chaingroup.chains:
                # choose a subset of the satisfied formulas and sample a state that satisfies them
                # pdb.set_trace()
                state = self._satisfy_subset(chain)
                # update chain counts
                chain.update(state)
            if self.verbose:
                bar.inc()
                bar.label('%d / %d' % (self.step, self.maxsteps))
            # intermediate results
            self.step += 1
        # get results
        self.step -= 1
        results = chaingroup.results()
        return results[0]

    def _satisfy_subset(self, chain):
        """
        Choose a set of logical formulas M to be satisfied (more specifically, M is a set of clause indices)
        and also choose a set of non-logical constraints NLC to satisfy
        """
        M = []   # must be satisfied by the next sampled state of the world
        NLC = []
        # pdb.set_trace()
        for gf_index, gf in enumerate(self.ground_formulas):
            literals = []
            if gf(chain.state) == 1 or gf.is_hard:
                exp_weight = math.exp(gf.weight)
                u = random.uniform(0, exp_weight)
                if hasattr(gf, "children"):
                    literals = [lt.ground_atom for lt in gf.children]
                if u > 1:
                    if gf.islogical():
                        # clause_range = self.gf2clauseindex[gf_index]
                        if len(literals) > 0:
                            for index in active_variables:
                                if self.mrf.variables[index].ground_atoms[0] in literals:
                                    active_clauses.append(gf_index)
                        clause_range = self.gf2clauseindex.setdefault(gf_index)
                        if clause_range is not None:  # and gf_index in active_clauses:
                            M.extend(list(range(*clause_range)))
                    else:
                        NLC.append(gf)
        # pdb.set_trace()
        # (uniformly) sample a state that satisfies them
        return SampleSAT(self.mrf, chain.state, M, NLC, self, p=self.p).run()

    def _prob_constraints_deviation(self):
        if len(self.softevidence) == 0:
            return {}
        se_mean, se_max, se_max_item = 0.0, -1, None
        for se in self.softevidence:
            dev = abs((se["numTrue"] / self.step) - se["p"])
            se_mean += dev
            if dev > se_max:
                se_max = max(se_max, dev)
                se_max_item = se
        se_mean /= len(self.softevidence)
        return {"pc_dev_mean": se_mean, "pc_dev_max": se_max, "pc_dev_max_item": se_max_item["expr"]}

    def _extend_results_history(self, results):
        cur_results = {"step": self.step, "results": list(results), "time": self._getElapsedTime()[0]}
        cur_results.update(self._getProbConstraintsDeviation())
        if self.referenceResults is not None:
            cur_results.update(self._compareResults(results, self.referenceResults))
        self.history.append(cur_results)

    def getResultsHistory(self):
        return self.resultsHistory


class SampleSAT:
    """
    Sample-SAT algorithm.
    """

    def __init__(self, mrf, state, clause_indices, nlcs, infer, p=1):
        """
        clause_indices: list of indices of clauses to satisfy
        p: probability of performing a greedy WalkSAT move
        state: the state (array of booleans) to work with (is reinitialized randomly by this constructor)
        NLConstraints: list of grounded non-logical constraints
        """
        self.debug = logger.level == logs.DEBUG
        self.infer = infer
        self.mrf = mrf
        self.mln = mrf.mln
        self.p = p
        # initialize the state randomly (considering the evidence) and obtain block info
        self.blockInfo = {}
        # self.state = self.infer.random_world()
        self.state = self.infer.smart_random_world()
        # out(self.state, '(initial state)')
        # pdb.set_trace()
        self.init = list(state)
        # these are the variables we need to consider for SampleSAT
        #         self.variables = [v for v in self.mrf.variables if v.valuecount(self.mrf.evidence) > 1]
        # list of unsatisfied constraints
        self.unsatisfied = set()
        # keep a map of bottlenecks: index of the ground atom -> list of constraints where the corresponding,
        # lit is a bottleneck
        self.bottlenecks = defaultdict(list)  # bottlenecks are clauses with exactly one true literal
        # ground atom occurrences in constraints: ground atom index -> list of constraints
        self.var2clauses = defaultdict(set)
        self.clauses = {}
        # instantiate clauses        
        for c_index in clause_indices:
            clause = SampleSAT._Clause(self.infer.clauses[c_index], self.state, c_index, self.mrf)
            self.clauses[c_index] = clause
            if clause.unsatisfied:
                self.unsatisfied.add(c_index)
            for v in clause.variables():
                self.var2clauses[v].add(clause)

        # instantiate non-logical constraints
        for nlc in nlcs:
            if isinstance(nlc, Logic.GroundCountConstraint):  # count constraint
                SampleSAT._CountConstraint(self, nlc)
            else:
                raise Exception("SampleSAT cannot handle constraints of type '%s'" % str(type(nlc)))

    def _print_unsatisfied_constraints(self):
        out("   %d unsatisfied:  %s" % (
            len(self.unsatisfied), list(map(str, [self.clauses[i] for i in self.unsatisfied]))), tb=2)

    def run(self):
        # sampling by enumerating all worlds, exact sampling, very slow
        """
        worlds = []
        count = 1
        for world in self.mrf.worlds():
            # if count > 1000:
            # break
            count += 1
            skip = False
            # pdb.set_trace()
            for clause in list(self.clauses.values()):
                if not clause.satisfied_in_world(world):
                    skip = True
                    break
            if skip:
                continue
            worlds.append(world)
        state = worlds[random.randint(0, len(worlds) - 1)]
        return state
        """
        # sample worlds, fast
        steps = 0
        # pdb.set_trace()
        while self.unsatisfied:
            steps += 1
            # make a WalkSat move or a simulated annealing move
            if random.uniform(0, 1) <= self.p:
                self._walksat_move()
            else:
                self._sa_move()
        return self.state

    def _walksat_move(self):
        """
        Randomly pick one of the unsatisfied constraints and satisfy it
        (or at least make one step towards satisfying it
        """
        # pdb.set_trace()
        clauseindex = list(self.unsatisfied)[random.randint(0, len(self.unsatisfied) - 1)]
        # get the literal that makes the fewest other formulas false
        clause = self.clauses[clauseindex]
        varval_opt = []
        opt = None
        for var in clause.variables():
            if var.index not in active_variables:
                continue
            bottleneck_clauses = [cl for cl in self.var2clauses[var] if cl.bottleneck is not None]
            for _, value in var.iter_values(self.mrf.evidence_dicti()):
                if not clause.turns_true_with(var, value):
                    continue
                unsat = 0
                for c in bottleneck_clauses:
                    # count the  constraints rendered unsatisfied for this value from the bottleneck atoms
                    turnsfalse = 1 if c.turns_false_with(var, value) else 0
                    unsat += turnsfalse
                append = False
                if opt is None or unsat < opt:
                    opt = unsat
                    varval_opt = []
                    append = True
                elif opt == unsat:
                    append = True
                if append:
                    varval_opt.append((var, value))
        if varval_opt:
            varval = varval_opt[random.randint(0, len(varval_opt) - 1)]
            self._setvar(*varval)

    def _setvar(self, var, val):
        """
        Set the truth value of a variable and update the information in the constraints.
        """
        var.setval(val, self.state)
        for c in self.var2clauses[var]:
            satisfied, _ = c.update(var, val)
            if satisfied:
                if c.c_index in self.unsatisfied: 
                    self.unsatisfied.remove(c.c_index)
            else:
                self.unsatisfied.add(c.c_index)

    def _sa_move(self):
        # randomly pick a variable and flip its value
        variables = list(set(self.var2clauses))
        random.shuffle(variables)
        var = variables[0]
        ev = var.evidence_value()
        values = var.value_count(self.mrf.evidence)
        for _, v in var.iter_values(self.mrf.evidence):
            break
        if values == 1:
            raise Exception('Only one remaining value for variable %s: %s. Please check your evidences.' % (var, v))
        values = [v for _, v in var.iter_values(self.mrf.evidence) if v != ev]
        val = values[random.randint(0, len(values) - 1)]
        unsat = 0
        bottleneck_clauses = [c for c in self.var2clauses[var] if c.bottleneck is not None]
        for c in bottleneck_clauses:
            # count the  constraints rendered unsatisfied for this value from the bottleneck clauses
            uns = 1 if c.turns_false_with(var, val) else 0
            #             cur = 1 if c.unsatisfied else 0
            unsat += uns  # - cur
        if unsat <= 0:  # the flip causes an improvement. take it with p=1.0
            p = 1.
        else:
            # !!! the temperature has a great effect on the uniformity of the sampled states! it's a "magic" number 
            # that needs to be chosen with care. if it's too low, then probabilities will be way off; if it's too high,
            # it will take longer to find solutions
            # temp = 14.0 # the higher the temperature, the greater the probability of deciding for a flip
            # we take as a heuristic the normalized difference between bottleneck clauses and clauses that actually
            # will turn false, such that if no clause will turn false, p is 1
            p = math.exp(- 1 + float(len(bottleneck_clauses) - unsat) / len(bottleneck_clauses))
        # decide and set
        if random.uniform(0, 1) <= p:
            self._setvar(var, val)

    class _Clause(object):

        def __init__(self, literals, world, index, mrf):
            self.c_index = index
            self.world = world
            self.bottleneck = None
            self.mrf = mrf
            # check all the literals
            self.literals = literals
            self.true_literals = set()
            self.atom_index2literals = defaultdict(set)
            for lit in literals:
                if isinstance(lit, Logic.TrueFalse):
                    continue
                atom_index = lit.ground_atom.index
                self.atom_index2literals[atom_index].add(0 if lit.negated else 1)
                if lit(world) == 1:
                    self.true_literals.add(atom_index)
            if len(self.true_literals) == 1 and self._is_bottleneck(item(self.true_literals)):
                self.bottleneck = item(self.true_literals)

        def _is_bottleneck(self, atom_index):
            atom_index2literals = self.atom_index2literals
            if len(self.true_literals) != 1 or atom_index not in self.true_literals:
                return False
            if len(atom_index2literals[atom_index]) == 1:
                return True
            fst = item(atom_index2literals[atom_index])
            if all([x == fst for x in atom_index2literals[atom_index]]): 
                return False  # the atom appears with different polarity in the clause, this is not a bottleneck
            return True

        def turns_false_with(self, var, val):
            """
            Returns whether or not this clause would become false if the given variable would take
            the given value. Returns False if the clause is already False.
            """
            for a, v in var.atom_values(val):
                if a.index == self.bottleneck and v not in self.atom_index2literals[a.index]:
                    return True
            return False

        def turns_true_with(self, var, val):
            """
            Returns true if this clause will be rendered true by the given variable taking
            its given value.
            """
            for a, v in var.atom_values(val):
                if self.unsatisfied and v in self.atom_index2literals[a.index]:
                    return True
            return False

        def update(self, var, val):
            """
            Updates the clause information with the given variable and value set in a SampleSAT state.
            """
            for a, v in var.atom_values(val):
                if v not in self.atom_index2literals[a.index]:
                    if a.index in self.true_literals: 
                        self.true_literals.remove(a.index)
                else:
                    self.true_literals.add(a.index)
            if len(self.true_literals) == 1 and self._is_bottleneck(item(self.true_literals)):
                self.bottleneck = item(self.true_literals)
            else:
                self.bottleneck = None
            return self.satisfied, self.bottleneck

        def satisfied_in_world(self, world):
            return self.mrf.mln.logic.disjugate(self.literals)(world) == 1

        @property
        def unsatisfied(self):
            return not self.true_literals

        @property
        def satisfied(self):
            return not self.unsatisfied

        def variables(self):
            return [self.mrf.variable(self.mrf.ground_atom(a)) for a in self.atom_index2literals]

        def greedySatisfy(self):
            self.ss._pickAndFlipLiteral([x.ground_atom.index for x in self.literals], self)

        def __str__(self):
            return ' v '.join(map(str, self.literals))

    class _CountConstraint:
        def __init__(self, sampleSAT, groundCountConstraint):
            self.ss = sampleSAT
            self.cc = groundCountConstraint
            self.trueOnes = []
            self.falseOnes = []
            # determine true and false ones
            for ga in groundCountConstraint.ground_atoms:
                indexGA = ga.index
                if self.ss.state[indexGA]:
                    self.trueOnes.append(indexGA)
                else:
                    self.falseOnes.append(indexGA)
                self.ss._addGAOccurrence(indexGA, self)
            # determine bottlenecks
            self._addBottlenecks()
            # if the formula is unsatisfied, add it to the list
            if not self._isSatisfied():
                self.ss.unsatisfiedConstraints.append(self)

        def _isSatisfied(self):
            return eval("len(self.trueOnes) %s self.cc.count" % self.cc.op)

        def _addBottlenecks(self):
            # there are only bottlenecks if we are at the border of the interval
            numTrue = len(self.trueOnes)
            if self.cc.op == "!=":
                trueNecks = numTrue == self.cc.count + 1
                falseNecks = numTrue == self.cc.count - 1
            else:
                border = numTrue == self.cc.count
                trueNecks = border and self.cc.op in ["==", ">="]
                falseNecks = border and self.cc.op in ["==", "<="]
            if trueNecks:
                for indexGA in self.trueOnes:
                    self.ss._addBottleneck(indexGA, self)
            if falseNecks:
                for indexGA in self.falseOnes:
                    self.ss._addBottleneck(indexGA, self)

        def greedySatisfy(self):
            c = len(self.trueOnes)
            satisfied = self._isSatisfied()
            assert not satisfied
            if c < self.cc.count and not satisfied:
                self.ss._pickAndFlipLiteral(self.falseOnes, self)
            elif c > self.cc.count and not satisfied:
                self.ss._pickAndFlipLiteral(self.trueOnes, self)
            else:  # count must be equal and op must be !=
                self.ss._pickAndFlipLiteral(self.trueOnes + self.falseOnes, self)

        def flipSatisfies(self, indexGA):
            if self._isSatisfied():
                return False
            c = len(self.trueOnes)
            if indexGA in self.trueOnes:
                c2 = c - 1
            else:
                assert indexGA in self.falseOnes
                c2 = c + 1
            return eval("c2 %s self.cc.count" % self.cc.op)

        def handleFlip(self, indexGA):
            """
            Handle all effects of the flip except bottlenecks of the flipped
            gnd atom and clauses that became unsatisfied as a result of a bottleneck flip
            """
            wasSatisfied = self._isSatisfied()
            # update true and false ones
            if indexGA in self.trueOnes:
                self.trueOnes.remove(indexGA)
                self.falseOnes.append(indexGA)
            else:
                self.trueOnes.append(indexGA)
                self.falseOnes.remove(indexGA)
            isSatisfied = self._isSatisfied()
            # if the constraint was previously satisfied and is now unsatisfied or
            # if the constraint was previously satisfied and is still satisfied (i.e. we are pushed further into the satisfying interval, away from the border),
            # remove all the bottlenecks (if any)
            if wasSatisfied:
                for indexground_atom in self.trueOnes + self.falseOnes:
                    if indexground_atom in self.ss.bottlenecks and self in self.ss.bottlenecks[
                        indexground_atom]:  # TODO perhaps have a smarter method to know which ones actually were bottlenecks (or even info about whether we had bottlenecks)
                        if indexGA != indexground_atom:
                            self.ss.bottlenecks[indexground_atom].remove(self)
                # the constraint was added to the list of unsatisfied ones in SampleSAT._flipground_atom (bottleneck flip)
            # if the constraint is newly satisfied, remove it from the list of unsatisfied ones
            elif not wasSatisfied and isSatisfied:
                self.ss.unsatisfiedConstraints.remove(self)
            # bottlenecks must be added if, because of the flip, we are now at the border of the satisfying interval
            self._addBottlenecks()

        def __str__(self):
            return str(self.cc)

        def getFormula(self):
            return self.cc


