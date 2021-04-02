"""
common utilities
"""

import itertools

from baize.ground.logic import GroundedAtom


def ground_predicate(predicate, domain_instances):
    """
    grounding the predicate
    """
    arg_instances = [domain_instances[arg.type] for arg in predicate.args]
    for params in itertools.product(*arg_instances):
        terms = list(params)
        yield GroundedAtom(predicate, terms)


def read_data(paths, predicate):
    content = []
    base_path = os.getcwd()
    file_ = open(base_path + '/' + paths, 'r')
    pre_content = file_.read()
    pre_content = pre_content.split('###')
    pre_content = [x for x in pre_content if x != '']
    for i in pre_content:
        element = i.split('\n')
        element = [x.replace(':', '_') for x in element if x != '']
        for j in element[1::]:
            splited = j.split('(')
            content.append((element[0], splited[0] + '(' + splited[1].upper()))

    return content

predicate = []
database = read_data("data.txt", predicate)
for i in enumerate(database):
    print i[1], i[1][1]
    for j in database[i[0]::]:
        print j[1]
