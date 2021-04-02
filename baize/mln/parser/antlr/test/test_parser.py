"""
test mln parser
"""

from baize.mln.parser.antlr.mln_parsers import parse_mln, parse_evidences, parse_queries


def test_parser():
    """
    test parser successful
    """

    program_str = """
    
    // Predicate definitions 
    * Friends(person, person)
    Smokes(person) 
    Cancer(person)
    
    // Rule definitions
    
    0.5 !Smokes(a1) , Cancer(a1) => Smokes(a1)
    0.5 !Smokes(a1) v Cancer(a1)
    0.4 !Friends(a1,a2) v !Smokes(a1) v Smokes(a2)
    0.4 !Friends(a1,a2) v !Smokes(a2) v Smokes(a1)
    """

    program = parse_mln(program_str)
    assert len(program._schemas) == 3
    pred_names = tuple(x.name for x in program.predicates())
    assert pred_names == ("Friends", "Smokes", "Cancer")

    assert len(program._rules) == 4
    rule_weights = tuple(x.weight for x in program.rules())
    assert rule_weights == (0.5, 0.5, 0.4, 0.4)

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

    queries_str = """
    Cancer(x)
    """

    queries = parse_queries(program, queries_str)
    assert len(queries) == 1


if __name__ == "__main__":
    test_parser()
