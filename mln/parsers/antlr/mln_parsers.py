"""
parse mln program
"""

import sys
from antlr4 import *
from .impl.MLNParser import MLNParser
from .impl.MLNLexer import MLNLexer
from .ast_visitors import NetworkListener, EvidencesListener, QueriesListener


def parse_mln(input_stream):
    """
    parse_mln_program
    """
    input = InputStream(input_stream)
    lexer = MLNLexer(input)
    stream = CommonTokenStream(lexer)
    parser = MLNParser(stream)
    tree = parser.definitions()

    mln_builder = NetworkListener()
    walker = ParseTreeWalker()
    walker.walk(mln_builder, tree)

    return mln_builder.mln


def parse_evidences(mln_program, input_stream):
    """
    parse_mln_program
    """
    input = InputStream(input_stream)
    lexer = MLNLexer(input)
    stream = CommonTokenStream(lexer)
    parser = MLNParser(stream)
    tree = parser.evidenceList()

    evidence_builder = EvidencesListener(mln_program)
    walker = ParseTreeWalker()
    walker.walk(evidence_builder, tree)

    return evidence_builder.evidences


def parse_queries(mln_program, input_stream):
    """
    parse_mln_program
    """
    input = InputStream(input_stream)
    lexer = MLNLexer(input)
    stream = CommonTokenStream(lexer)
    parser = MLNParser(stream)
    tree = parser.queryList()

    query_builder = QueriesListener(mln_program)
    walker = ParseTreeWalker()
    walker.walk(query_builder, tree)

    return query_builder.queries
