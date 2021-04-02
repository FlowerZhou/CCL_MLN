from mln.inference.gibbs import GibbsSampler
from mln.inference.exact import EnumerationAsk
from mln.inference.mcsat import MCSAT
from mln.learning.cll import CLL, DCLL
from mln.learning.ll import LL
from mln.learning.bpll import BPLL, DPLL, BPLL_CG, DBPLL_CG
from mln.inference.maxwalk import SAMaxWalkSAT


class Enum(object):

    def __init__(self, items):
        self.id2name = dict([(clazz.__name__, name) for (clazz, name) in items])
        self.name2id = dict([(name, clazz.__name__) for (clazz, name) in items])
        self.id2clazz = dict([(clazz.__name__, clazz) for (clazz, _) in items])

    def __getattr__(self, id_):
        if id_ in self.id2clazz:
            return self.id2clazz[id_]
        raise KeyError('Enum does not define %s, only %s' % (id_, self.id2clazz.keys()))

    def clazz(self, key):
        if type(key).__name__ == 'type':
            key = key.__name__
        if key in self.id2clazz:
            return self.id2clazz[str(key)]
        else:
            return self.id2clazz[self.name2id[key]]
        raise KeyError('No such element "%s"' % key)

    def id(self, key):
        if type(key).__name__ == 'type':
            return key.__name__
        if key in self.name2id:
            return self.name2id[key]
        raise KeyError('No such element "%s"' % key)

    def name(self, id_):
        if id_ in self.id2name:
            return self.id2name[id_]
        raise KeyError('No element with id "%s"' % id_)

    def names(self):
        return self.id2name.values()

    def ids(self):
        return self.id2name.keys()


InferenceMethods = Enum(
    (
        (GibbsSampler, 'Gibbs sampling'),
        (EnumerationAsk, 'Enumeration-Ask (exact)'),
        (MCSAT, 'MC-SAT'),
        (SAMaxWalkSAT, 'Max-Walk-SAT with simulated annealing (approx. MPE)')

    ))

LearningMethods = Enum(
    (
        (CLL, 'composite-log-likelihood'),
        (DCLL, '[discriminative] composite-log-likelihood'),
        (LL, "log-likelihood"),
        (DPLL, '[discriminative] pseudo-log-likelihood'),
        (BPLL, 'pseudo-log-likelihood'),
        (BPLL_CG, 'pseudo-log-likelihood (fast conjunction grounding)'),
        (DBPLL_CG, '[discriminative] pseudo-log-likelihood (fast conjunction grounding)')

    ))

if __name__ == '__main__':
    print(InferenceMethods.id2clazz)
    print(InferenceMethods.id2name)
    print(InferenceMethods.name2id)
    print(LearningMethods.names())
