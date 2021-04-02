"""
naive grounding algorithms
"""
import itertools
from collections import defaultdict
from typing import List

from baize.ground.logic import GroundedClause, TruthValue, predicate_voc, constant_voc
from baize.logic import Evidence
from baize.mln.markov_logic_network import MarkovLogicNet

from loguru import logger


class NaiveGrounder(object):
    """
    NaiveGrounder
    """

    def __init__(self, mln: MarkovLogicNet, evidences: List[Evidence]):
        self.mln = mln
        self.evidences = evidences

        self.domain_instances = defaultdict(set)

        for evidence in self.evidences:
            pred_name = predicate_voc[evidence.atom.predicate]
            predicate = self.mln.get_predicate(pred_name)
            for arg, instance in zip(predicate.args, evidence.atom.terms):
            # logger.debug("domain {0} has instance {1}".format(arg.type, constant_voc[instance]))
                self.domain_instances[arg.type].add(instance)

        self.evidence_truth = dict((x.atom, x.truth) for x in self.evidences)
        logger.debug("domains :" + str(self.domain_instances.keys()))

    def ground_clause(self, clause):
        """
        grounding the clause
        """
        var_instances = defaultdict(set)

        for literal in clause.literals:
            for arg, term in zip(literal.atom.predicate.args, literal.atom.terms):
                if term.is_var:
                    var_instances[term.name].update(self.domain_instances[arg.type])

        variables = list(var_instances.keys())
        instances = [var_instances[x] for x in variables]
        logger.debug("vars = " + str(variables))
        for one_instance in itertools.product(* instances):
            logger.debug("instance = " + str(one_instance))
            assign = dict(zip(variables, one_instance))

            g_clause = GroundedClause.grounded_from(clause, assign)

            logger.debug("grounding clause: " + str(g_clause))

            yield g_clause

    def num_groundings(self, clause):

        grounding_num = 0
        for g_clause in self.ground_clause(clause):
            truth = g_clause.truth(self.evidence_truth)
            if truth == TruthValue.TRUE:
                grounding_num += 1

        return grounding_num



