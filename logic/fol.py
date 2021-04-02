from dnutils import ifnone

from .elements import Logic
from mln.util import fstr


class FirstOrderLogic(Logic):
    """
    Factory class for first-order logic.
    """

    class Constraint(Logic.Constraints):
        pass

    class Formula(Logic.Formula, Constraint):

        def noisy_or(self, world):
            """
            Computes the noisy-or distribution of this formula.
            """
            return self.cnf().noisy_or(world)

        def _get_evidence_truth_degree_cw(self, ground_atom, world_values):
            """
                gets (soft or hard) evidence as a degree of belief from 0 to 1, making the closed world assumption,
                soft evidence has precedence over hard evidence
            """
            se = self._getSoftEvidence(ground_atom)
            if se is not None:
                return se if (world_values[ground_atom.index] or world_values is None[
                    ground_atom.index]) else 1.0 - se  # TODO allSoft currently unsupported
            return 1.0 if world_values[ground_atom.index] else 0.0

        def _noisy_or(self, mln, world_values, disj):
            if isinstance(disj, FirstOrderLogic.GroundLiteral):
                literals = [disj]
            elif isinstance(disj, FirstOrderLogic.TrueFalse):
                return disj.isTrue(world_values)
            else:
                literals = disj.children
            prod = 1.0
            for lit in literals:
                p = mln._get_evidence_truth_degree_cw(lit.ground_atom, world_values)
                if not lit.negated:
                    factor = p
                else:
                    factor = 1.0 - p
                prod *= 1.0 - factor
            return 1.0 - prod

    class ComplexFormula(Logic.ComplexFormula, Formula):
        pass

    class Literal(Logic.Literal, Formula):
        pass

    class LiteralGroup(Logic.LiteralGroup, Formula):
        pass

    class GroundAtom(Logic.GroundAtom):
        pass

    class GroundLiteral(Logic.GroundLiteral, Formula):

        def noisy_or(self, world):
            truth = self(world)
            if self.negated:
                truth = 1. - truth
            return truth

    class Disjunction(Logic.Disjunction, ComplexFormula):

        def truth(self, world):
            dont_know = False
            for child in self.children:
                child_value = child.truth(world)
                if child_value == 1:
                    return 1
                if child_value is None:
                    dont_know = True
            if dont_know:
                return None
            else:
                return 0

        def simplify(self, world):
            sf_children = []
            for child in self.children:
                child = child.simplify(world)
                t = child.truth(world)
                if t == 1:
                    return self.mln.logic.true_false(1, mln=self.mln, index=self.index)
                elif t == 0:
                    continue
                else:
                    sf_children.append(child)
            if len(sf_children) == 1:
                return sf_children[0].copy(index=self.index)
            elif len(sf_children) >= 2:
                return self.mln.logic.disjunction(sf_children, mln=self.mln, index=self.index)
            else:
                return self.mln.logic.true_false(0, mln=self.mln, index=self.index)

        def noisy_or(self, world):
            prod = 1.0
            for lit in self.children:
                p = ifnone(lit(world), 1)
                if not lit.negated:
                    factor = p
                else:
                    factor = 1.0 - p
                prod *= 1.0 - factor
            return 1.0 - prod

    class Conjunction(Logic.Conjunction, ComplexFormula):

        def truth(self, world):
            dont_know = False
            for child in self.children:
                child_value = child.truth(world)
                if child_value == 0:
                    return 0.
                if child_value is None:
                    dont_know = True
            if dont_know:
                return None
            else:
                return 1.

        def simplify(self, world):
            sf_children = []
            for child in self.children:
                child = child.simplify(world)
                t = child.truth(world)
                if t == 0:
                    return self.mln.logic.true_false(0, mln=self.mln, index=self.index)
                elif t == 1:
                    pass
                else:
                    sf_children.append(child)
            if len(sf_children) == 1:
                return sf_children[0].copy(index=self.index)
            elif len(sf_children) >= 2:
                return self.mln.logic.conjunction(sf_children, mln=self.mln, index=self.index)
            else:
                return self.mln.logic.true_false(1, mln=self.mln, index=self.index)

        def noisy_or(self, world):
            cnf = self.cnf()
            prod = 1.0
            if isinstance(cnf, FirstOrderLogic.Conjunction):
                for disj in cnf.children:
                    prod *= disj.noisyor(world)
            else:
                prod *= cnf.noisyor(world)
            return prod

    class Implication(Logic.Implication, ComplexFormula):

        def truth(self, world):
            ant = self.children[0].truth(world)
            cons = self.children[1].truth(world)
            if ant == 0 or cons == 1:
                return 1
            if ant is None or cons is None:
                return None
            return 0

    class BiImplication(Logic.BiImplication, ComplexFormula):

        def truth(self, world):
            c1 = self.children[0].truth(world)
            c2 = self.children[1].truth(world)
            if c1 is None or c2 is None:
                return None
            return 1 if (c1 == c2) else 0

    class Negation(Logic.Negation, ComplexFormula):
        pass

    class Exist(Logic.Exist, ComplexFormula):
        pass

    class Equality(Logic.Equality, ComplexFormula):
        pass

    class TrueFalse(Logic.TrueFalse, Formula):

        @property
        def value(self):
            return self._value

        @value.setter
        def value(self, truth):
            if not truth == 0 and not truth == 1:
                raise Exception('Truth values in first-order logic cannot be %s' % truth)
            self._value = truth

        def __str__(self):
            return str(True if self.value == 1 else False)

        def noisy_or(self, world):
            return self(world)

    class ProbabilityConstraint(object):
        """
        Base class for representing a prior/posterior probability constraint (soft evidence)
        on a logical expression.
        """

        def __init__(self, formula, p):
            self.formula = formula
            self.p = p

        def __repr__(self):
            return str(self)

    class PriorConstraint(ProbabilityConstraint):
        """
        Class representing a prior probability.
        """

        def __str__(self):
            return 'P(%s) = %.2f' % (fstr(self.formula), self.p)

    class PosteriorConstraint(ProbabilityConstraint):
        """
        Class representing a posterior probability.
        """

        def __str__(self):
            return 'P(%s|E) = %.2f' % (fstr(self.formula), self.p)

    def conjunction(self, *args, **kwargs):
        return FirstOrderLogic.Conjunction(*args, **kwargs)

    def disjunction(self, *args, **kwargs):
        return FirstOrderLogic.Disjunction(*args, **kwargs)

    def negation(self, *args, **kwargs):
        return FirstOrderLogic.Negation(*args, **kwargs)

    def implication(self, *args, **kwargs):
        return FirstOrderLogic.Implication(*args, **kwargs)

    def bi_implication(self, *args, **kwargs):
        return FirstOrderLogic.BiImplication(*args, **kwargs)

    def equality(self, *args, **kwargs):
        return FirstOrderLogic.Equality(*args, **kwargs)

    def exist(self, *args, **kwargs):
        return FirstOrderLogic.Exist(*args, **kwargs)

    def ground_atom(self, *args, **kwargs):
        return FirstOrderLogic.GroundAtom(*args, **kwargs)

    def literal(self, *args, **kwargs):
        return FirstOrderLogic.Literal(*args, **kwargs)

    def literal_group(self, *args, **kwargs):
        return FirstOrderLogic.LiteralGroup(*args, **kwargs)

    def ground_literal(self, *args, **kwargs):
        return FirstOrderLogic.GroundLiteral(*args, **kwargs)

    def count_constraint(self, *args, **kwargs):
        return FirstOrderLogic.CountConstraint(*args, **kwargs)

    def true_false(self, *args, **kwargs):
        return FirstOrderLogic.TrueFalse(*args, **kwargs)


# this is a little hack to make nested classes pickle-able
Constraint = FirstOrderLogic.Constraint
Formula = FirstOrderLogic.Formula
ComplexFormula = FirstOrderLogic.ComplexFormula
Conjunction = FirstOrderLogic.Conjunction
Disjunction = FirstOrderLogic.Disjunction
Literal = FirstOrderLogic.Literal
GroundLiteral = FirstOrderLogic.GroundLiteral
GroundAtom = FirstOrderLogic.GroundAtom
Equality = FirstOrderLogic.Equality
Implication = FirstOrderLogic.Implication
BiImplication = FirstOrderLogic.BiImplication
Negation = FirstOrderLogic.Negation
Exist = FirstOrderLogic.Exist
TrueFalse = FirstOrderLogic.TrueFalse
NonLogicalConstraint = FirstOrderLogic.NonLogicalConstraint
CountConstraint = FirstOrderLogic.CountConstraint
GroundCountConstraint = FirstOrderLogic.GroundCountConstraint
