"""
test grounding
"""
from baize.ground.naive_grounder import NaiveGrounder
from baize.mln.parser.antlr.mln_parsers import parse_mln, parse_evidences

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


def test_naive_grounding():

    mln = parse_mln(program_str)

    evidences = parse_evidences(mln, evidences_str)

    grounder = NaiveGrounder(mln, evidences)

    for clause in mln.rules():
        num_of_grounding = grounder.num_groundings(clause)
        print(num_of_grounding)

