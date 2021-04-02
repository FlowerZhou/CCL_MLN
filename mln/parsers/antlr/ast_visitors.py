"""
listener: visitor for ast tree
"""
import sys
import pdb
import re
import traceback
sys.path.append('../../../../../../')
from ground.logic import GroundedAtom
from baize.logic import PredicateArgument, Term, Literal, Clause, Atom, Evidence
from mln.mlnpreds import Predicate
from .impl.MLNListener import MLNListener
from .impl.MLNParser import MLNParser
from mln.markov_logic_network import MarkovLogicNet


class NetworkListener(MLNListener):
    """
    MLNProgramListener
    """

    def __init__(self):
        self.mln = MarkovLogicNet()

    def exitPredArg(self, ctx: MLNParser.PredArgContext):
        """
        exitPredArg
        """
        # pdb.set_trace()
        arg = PredicateArgument()
        arg.type = ctx.type_.text
        if ctx.name:
            arg.name = ctx.name.text
        if ctx.uni:
            arg.unique = ctx.uni.text == "!"

        ctx.arg = arg

    def exitSchema(self, ctx: MLNParser.SchemaContext):
        """
        exitSchema
        """
        # pdb.set_trace()
        # predicate = Predicate()
        # predicate.name = ctx.pname.text
        # predicate.args = [x.arg for x in ctx.types]
        pred_name = ctx.pname.text
        argdoms = [x.arg for x in ctx.types]
        arg_doms = []
        for arg in argdoms:
            arg_doms.append(arg.type)
        # if ctx.a1:
            # predicate.closed_world = ctx.a1.text == "*"
        print(type(arg_doms))
        print(type(arg_doms[0]))
        predicate = Predicate(pred_name, arg_doms)

        self.mln.predicate(predicate)

    def exitTerm(self, ctx: MLNParser.TermContext):
        """
        exitTerm
        """

        term = Term()
        if ctx.x:
            term.name = ctx.x.text
            term.is_var = not term.name[0].isupper()  # if id[0].isupper(), it is a constant
        elif ctx.d:
            term.name = ctx.x.text
            term.is_var = False

        ctx.term = term

    def exitAtom(self, ctx: MLNParser.AtomContext):
        """
        exit Atom
        """
        # pdb.set_trace()
        atom = Atom()
        pred_name = ctx.pred.text
        try:
            atom.predicate = self.mln.get_predicate(pred_name)
            # print(pred_name)
        except Exception as e:
            traceback.print_exc()
            for name in self.mln._schemas:
                print(name)
            raise Exception("Cannot find a predicate with name : " + pred_name)

        atom.terms = [x.term for x in ctx.terms]

        if len(atom.terms) != len(atom.predicate.arg_doms):
            raise Exception("{0} terms in the literal but expecting {1} args for the predicate {2}".format(
                len(atom.terms), len(atom.predicate.arg_doms), pred_name
            ))
        ctx.atom = atom

    def exitLiteral(self, ctx: MLNParser.LiteralContext):
        """
        exitLiteral
        """
        # pdb.set_trace()
        literal = Literal()
        literal.atom = ctx.a.atom

        if ctx.pref:
            if ctx.pref.text == "!":
                literal.sense = False
            elif ctx.pref.text == "+":
                pass

        ctx.literal = literal

    def exitFoclause(self, ctx: MLNParser.FoclauseContext):
        """
        exitFoclause
        """
        # pdb.set_trace()
        clause = Clause()

        for lit_ctx in ctx.ants:
            literal = lit_ctx.literal
            literal.flip_sense()
            clause.literals.append(literal)

        for lit_ctx in ctx.lits:
            clause.literals.append(lit_ctx.literal)

        if ctx.exq:
            for var in ctx.exq.vs:
                clause.existential_vars.append(var.text)

        ctx.clause = clause

    def exitHardRule(self, ctx: MLNParser.HardRuleContext):
        """
        exitHardRule
        """
        pdb.set_trace()
        clause = ctx.fc.clause
        clause.set_hard_weight()

        self.mln.add_rule(clause)

    def exitSoftRule(self, ctx: MLNParser.SoftRuleContext):
        """
        exitSoftRule
        """
        pdb.set_trace()
        clause = ctx.fc.clause
        if ctx.weight:
            clause.weight = float(ctx.weight.text)
        elif ctx.warg:
            clause.weight = ctx.warg.text

        print(ctx.getText())
        formula = re.sub("[0-9.]", "", ctx.getText())
        print(formula)
        if ctx.du:
            clause.fixed_weight = True
        else:
            clause.fixed_weight = False

        # self.mln.add_rule(clause)
        self.mln.formula(formula, clause.weight, fix_weight=False, unique_templ_vars=None)


class GroundedAtomListener(MLNListener):
    """
    GroundedAtomListener
    """

    def __init__(self, program):
        super().__init__()
        self.program = program

    def exitTerm(self, ctx: MLNParser.TermContext):
        """
        exitTerm
        """
        term = None
        if ctx.x:
            term = ctx.x.text
            # print(term)
            if not term[0].isupper():  # if id[0].isupper(), it is a constant

                raise Exception("Evidence must not contain variables: " + term)
        elif ctx.d:
            term = ctx.d.text

        ctx.term = term

    def exitAtom(self, ctx: MLNParser.AtomContext):
        """
        exit Atom
        """

        pred_name = ctx.pred.text
        try:
            predicate = self.program.get_predicate(pred_name)
        except Exception as e:
            traceback.print_exc()
            for name in self.program._schemas:
                print(name)
            raise Exception("Cannot find a predicate with name : " + pred_name)

        terms = [x.term for x in ctx.terms]
        atom = GroundedAtom(predicate, terms)
        if len(atom.terms) != len(predicate.args):
            raise Exception("{0} terms in the literal but expecting {1} args for the predicate {2}".format(
                len(atom.terms), len(predicate.args), pred_name
            ))
        ctx.atom = atom


class EvidencesListener(GroundedAtomListener):
    """
    MLNProgramListener
    """

    def __init__(self, program):
        super().__init__(program)

        self.evidences = []

    def exitEvidence(self, ctx:MLNParser.EvidenceContext):

        """
        exitLiteral
        """

        evidence = Evidence()
        evidence.atom = ctx.a.atom

        if ctx.prior:
            evidence.prior = float(ctx.prior.text)
            if ctx.perf:
                evidence.prior = 1 - evidence.prior
        else:
            evidence.truth = ctx.perf is None

        self.evidences.append(evidence)


class QueriesListener(GroundedAtomListener):
    """
    MLNProgramListener
    """

    def __init__(self, program):
        super().__init__(program)
        self.queries = []

    def exitQuery(self, ctx:MLNParser.QueryContext):

        """
        exitLiteral
        """

        atom = ctx.a.atom
        self.queries.append(atom)
