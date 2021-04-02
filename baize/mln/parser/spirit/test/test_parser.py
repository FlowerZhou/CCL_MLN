"""

"""

from baize.mln.parser.parser import parse_mln_program


def test_parse_successful():
    """
    test parser successful
    """
    """
      
    // Predicate definitions 
    * Friends(person, person)
    Smokes(person) 
    Cancer(person)
     
     """

    program_str = """
    // Rule definitions
    
    0.5 !Smokes(a1) , Cancer(a1) => Smokes(a1)
    0.5 !Smokes(a1) v Cancer(a1)
    0.4 !Friends(a1,a2) v !Smokes(a1) v Smokes(a2)
    0.4 !Friends(a1,a2) v !Smokes(a2) v Smokes(a1)
    """

    assert parse_mln_program(program_str)


if __name__ == "__main__":
    test_parse_successful()
