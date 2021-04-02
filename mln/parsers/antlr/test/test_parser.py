"""
test mln parser
"""

import sys
import os
sys.path.append("../../../../")
# sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from mln.parsers.antlr.mln_parsers import *
import pdb


def test_parser():
    """
    test parser successful
    """

    program_str = """
    
    // Predicate definitions 
    Friends(person, person)
    Smokes(person) 
    Cancer(person)
    
    
    """
    # pdb.set_trace()
    program = parse_mln(program_str)
    assert len(program._predicates) == 3
    pred_names = tuple(x.name for x in program.iter_predicates())
    print(pred_names)
    # assert pred_names == ("Friends", "Smokes", "Cancer")

    # assert len(program._rules) == 4
    # rule_weights = tuple(x.weight for x in program.rules())
    # assert rule_weights == (0.5, 0.5, 0.4, 0.4)

    evidences_str = """
    Friends(Anna, Bob) 
    Friends(Anna, Edward) 
    Friends(Anna, Frank) 
    Friends(Edward, Frank) 
    Friends(Gary, Helen) 
    !Friends(Gary, Frank) 
    Smokes(Anna) 
    Smokes(Edward)
    """

    evidences = parse_evidences(program, evidences_str)
    assert len(evidences) == 8
    # print(evidences)
    queries_str = """
    Cancer(Bob)
    """

    queries = parse_queries(program, queries_str)
    assert len(queries) == 1
    # print(queries)


if __name__ == "__main__":
    test_parser()

"""
// Rule definitions

    0.5 Cancer(x) => Smokes(x)
    0.5 !Smokes(a1) , Cancer(a1) => Smokes(a1)
    0.5 !Smokes(a1) v Cancer(a1)
    0.4 !Friends(a1,a2) v !Smokes(a1) v Smokes(a2)
    0.4 !Friends(a1,a2) v !Smokes(a2) v Smokes(a1)
"""