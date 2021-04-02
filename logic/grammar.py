from pyparsing import *
import re
import pdb


class TreeBuilder(object):
    """
    the parsing tree
    """
    def __init__(self, logic):
        self.logic = logic
        self.reset()

    def trigger(self, a, loc, toks, op):
        # print ("Enter Trigger !")
        # pdb.set_trace()

        if op == 'literal_group':
            negated = False
            if toks[0] == "!" or toks[0] == "*":
                if toks[0] == "*":
                    negated = 2
                else:
                    negated = True
                toks = toks[1]
            else:
                toks = toks[0]
            self.stack.append(self.logic.literal_group(negated, toks[:-1], toks[-1], self.logic.mln))
        if op == "literal":
            print("enter literal")
            negated = False
            if toks[0] == "!" or toks[0] == "*":
                if toks[0] == "*":
                    negated = 2
                else:
                    negated = True
                toks = toks[1]
            else:
                toks = toks[0]
            self.stack.append(self.logic.literal(negated, toks[0], toks[1], self.logic.mln))
        elif op == '!':
            if len(toks) == 1:
                formula = self.logic.negation(self.stack[-1:], self.logic.mln)
                self.stack = self.stack[:-1]
                self.stack.append(formula)
        elif op == '=':
            if len(toks) == 2:
                self.stack.append(self.logic.equality(list(toks), False, self.logic.mln))
        elif op == '!=':
            if len(toks) == 2:
                self.stack.append(self.logic.equality(list(toks), True, self.logic.mln))
        elif op == '^':
            if len(toks) > 1:
                formula = self.logic.conjunction(self.stack[-len(toks):], self.logic.mln)
                self.stack = self.stack[:-len(toks)]
                self.stack.append(formula)
        elif op == 'v':
            if len(toks) > 1:
                formula = self.logic.disjunction(self.stack[-len(toks):], self.logic.mln)
                self.stack = self.stack[:-len(toks)]

                self.stack.append(formula)
        elif op == '=>':
            # pdb.set_trace()
            if len(toks) == 2:
                children = self.stack[-2:]
                self.stack = self.stack[:-2]
                self.stack.append(self.logic.implication(children, self.logic.mln))
        elif op == '<=>':
            if len(toks) == 2:
                children = self.stack[-2:]
                self.stack = self.stack[:-2]
                self.stack.append(self.logic.bi_implication(children, self.logic.mln))
        elif op == 'ex':
            if len(toks) == 2:
                formula = self.stack.pop()
                variables = map(str, toks[0])
                self.stack.append(self.logic.exist(variables, formula, self.logic.mln))
        elif op == 'count':
            if len(toks) in (3, 4):
                pred, pred_params = toks[0]
                if len(toks) == 3:
                    fixed_params, op, count = [], toks[1], int(toks[2])
                else:
                    fixed_params, op, count = list(toks[1]), toks[2], int(toks[3])
                self.stack.append(self.logic.count_constraint(pred, pred_params, fixed_params, op, count))
        return self.stack[-1]

    def reset(self):
        self.stack = []

    def get_constraint(self):
        if len(self.stack) > 1:
            raise Exception("Not a valid formula - reduces to more than one element %s" % str(self.stack))
        if len(self.stack) == 0:
            raise Exception("Constraint could not be parsed")
            #  if not isinstance(self.stack[0], Logic.Constraint):
            #  raise Exception("Not an instance of Constraint!")
        return self.stack[0]


class Grammar(object):
    """
    abstract super class for all logic grammars
    """

    def __deepcopy__(self, memo):
        return self

    def parse_formula(self, s):
        self.tree.reset()
        self.formula.parseString(s)
        con_str = self.tree.get_constraint()
        return con_str

    def parse_atom(self, string):
        """
        Parses a predicate such as p(A,B) and returns a tuple where the first item
        is the predicate name and the second is a list of parameters, e.g. ("p", ["A", "B"])
        """
        m = re.match(r'(\w+)\((.*?)\)$', string)
        if m is not None:
            return m.group(1), map(str.strip, m.group(2).split(","))

    def parse_predicate(self, s):
        return self.pred_decl.parseString(s)[0]

    def is_var(self, identifier):
        raise Exception('%s does not implement is_var().' % str(type(self)))

    def is_constant(self, identifier):
        return not self.is_var(identifier)

    def is_templ_var(self, s):
        return s[0] == '+' and self.is_var(s[1:])

    def parse_domain(self, s):
        """
        parses a domain declaration and returns a tuple (domain name, list of constants)
        return none if it cannot be parsed
        """
        m = re.match(r'(\w+)\s*=\s*{(.*?)}', s)
        if m is None:
            return None
        return m.group(1), map(str.strip, m.group(2).split(','))

    def parse_literal(self, s):
        """
        Parses a literal such as !p(A,B) or p(A,B)=False and returns a tuple
        where the first item is whether the literal is true, the second is the
        predicate name and the third is a list of parameters, e.g. (False, "p", ["A", "B"])
        """
        self.tree.reset()
        literal = self.literal.parseString(s)
        literal = self.tree.get_constraint()
        return not literal.negated, literal.pred_name, literal.args


class StandardGrammar(Grammar):
    """
    the standard MLN logic syntax
    """
    def __init__(self, logic):
        identifier_character = alphanums + '_' + '-' + "'"
        lc_character = alphas.lower()
        uc_character = alphas.upper()
        lc_name = Word(lc_character, alphanums + '_')

        open_rb = Literal("(").suppress()
        close_rb = Literal(")").suppress()

        dom_name = Combine(Optional(Literal(':')) + lc_name + Optional(Literal('!') | Literal('?')))
        constant = Word(identifier_character) | Word(nums) | Combine(Literal('"') + Word(printables.replace('"', '')) + Literal('"'))
        variable = Word(lc_character, identifier_character)
        atom_args = Group(delimitedList(constant | Combine(Optional("+")+variable)))
        pred_decl_args = Group(delimitedList(dom_name))
        pred_name = Word(identifier_character)

        atom = Group(pred_name + open_rb + atom_args + close_rb)
        literal = Optional(Literal("!") | Literal("*")) + atom
        ground_atom_args = Group(delimitedList(constant))
        ground_literal = Optional(Literal("!")) + Group(pred_name + open_rb + ground_atom_args + close_rb)
        pred_decl = Group(pred_name + open_rb + pred_decl_args + close_rb) + StringEnd()

        var_list = Group(delimitedList(variable))
        count_constraint = Literal("count(").suppress() + atom + Optional(Literal("|").suppress() + var_list) + \
                           Literal(")").suppress() + (Literal("=") | Literal(">=") | Literal("<=")) + Word(nums)
        # print(" formula = forward!")
        formula = Forward()
        # print (" formula << constant")
        exist = Literal("EXIST").suppress() + Group(delimitedList(variable)) + open_rb + Group(formula) + close_rb
        equality = (constant | variable) + Literal("=").suppress() + (constant | variable)
        inequality = (constant | variable) + Literal("!=").suppress() + (constant | variable)
        negation = Literal("!").suppress() + open_rb + Group(formula) + close_rb
        item = literal | exist | equality | open_rb + formula + close_rb | negation
        disjunction = Group(item) + ZeroOrMore(Literal("v").suppress() + Group(item))
        conjunction = Group(disjunction) + ZeroOrMore(Literal("^").suppress() + Group(disjunction))
        implication = Group(conjunction) + Optional(Literal("=>").suppress() + Group(conjunction))
        bi_implication = Group(implication) + Optional(Literal("<=>").suppress() + Group(implication))
        constraint = bi_implication | count_constraint
        formula << constraint

        def literal_parse_action(a, b, c): tree.trigger(a, b, c, 'literal')
        def ground_literal_parse_action(a, b, c): tree.trigger(a, b, c, "ground_literal")
        def neg_parse_action(a, b, c): tree.trigger(a, b, c, '!')
        def disjunction_parse_action(a, b, c): tree.trigger(a, b, c, 'v')
        def conjunction_parse_action(a, b, c): tree.trigger(a, b, c, '^')
        def exist_parse_action(a, b, c): tree.trigger(a, b, c, "ex")
        def implication_parse_action(a, b, c): tree.trigger(a, b, c, "=>")
        def bi_implication_parse_action(a, b, c): tree.trigger(a, b, c, "<=>")
        def equality_parse_action(a, b, c): tree.trigger(a, b, c, "=")
        def inequality_parse_action(a, b, c): tree.trigger(a, b, c, "!=")
        def count_constraint_parse_action(a, b, c): tree.trigger(a, b, c, 'count')

        tree = TreeBuilder(logic)
        literal.setParseAction(literal_parse_action)
        ground_literal.setParseAction(ground_literal_parse_action)
        negation.setParseAction(neg_parse_action)
        disjunction.setParseAction(disjunction_parse_action)
        conjunction.setParseAction(conjunction_parse_action)
        exist.setParseAction(exist_parse_action)
        implication.setParseAction(implication_parse_action)
        bi_implication.setParseAction(bi_implication_parse_action)
        equality.setParseAction(equality_parse_action)
        inequality.setParseAction(inequality_parse_action)
        count_constraint.setParseAction(count_constraint_parse_action)

        self.tree = tree
        self.formula = formula + StringEnd()
        self.pred_decl = pred_decl
        self.literal = literal
        # print("Standard Grammar init complete")

    def is_var(self, identifier):
        return identifier[0].islower() or identifier[0] == '+'




















