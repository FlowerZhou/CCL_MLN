"""
build and encapsule
"""
import cppimport.import_hook
from baize.mln.parser.spirit import parser_impl

def parse_mln_program(program_str):
    """
    parse_mln_program
    """

    return parser_impl.parse_mln_program(program_str)