from dnutils import logs

from . import optimize
import sys
from numpy.ma.core import exp
from mln.constants import HARD
import numpy
import pdb
from numba import jit


logger = logs.getlogger(__name__)


class AbstractLearner(object):
    """
    Abstract base class for every MLN learning algorithm.
    """

    def __init__(self, mrf=None, **params):
        self.mrf = mrf
        self._params = params
        # pdb.set_trace()
        self.mrf.apply_cw()
        self._w = None

    @property
    def prior_stdev(self):
        """
        prior_stdev: The standard deviation of the prior on the weights
        """
        return self._params.get('prior_stdev')

    @property
    def verbose(self):
        return self._params.get('verbose', False)

    @property
    def use_init_weights(self):
        return self._params.get('use_init_weights')

    @property
    def usegrad(self):
        return True

    @property
    def usef(self):
        return True

    @property
    def multicore(self):
        return self._params.get('multicore', True)

    @property
    def weights(self):
        return self._w

    @property
    def maxrepeat(self):
        return self._params.get('maxrepeat', 1)

    def repeat(self):
        return False

    def _add_fix_weights(self, w):
        i = 0
        w_ = []
        for f in self.mrf.formulas:
            if self.mrf.mln.fix_weights[f.index] or f.weight == HARD:
                w_.append(self._w[f.index])
            else:
                w_.append(w[i])
                i += 1
        return w_

    def _var_weights(self):
        return self._filter_fix_weights(self._w)

    def f(self, weights):
        # reconstruct full weight vector
        w = self._add_fix_weights(weights)
        # compute prior
        prior = 0
        if self.prior_stdev is not None:
            for w_ in w:  # we have to use the log of the prior here
                prior -= 1. / (2. * (self.prior_stdev ** 2)) * w_ ** 2
                # compute log likelihood
        likelihood = self._f(w)
        if self.verbose:
            sys.stdout.write('                                           \r')
            if self.prior_stdev is not None:
                sys.stdout.write('  log P(D|w) + log P(w) = %f + %f = %f\r' % (likelihood, prior, likelihood + prior))
            else:
                sys.stdout.write('  log P(D|w) = %f\r' % likelihood)
            sys.stdout.flush()
        return likelihood + prior

    def grad(self, weights):
        w = self._add_fix_weights(weights)
        grad = self._grad(w)
        self._grad_ = grad
        # add gaussian prior
        if self.prior_stdev is not None:
            for i, weight in enumerate(w):
                grad[i] -= 1. / (self.prior_stdev ** 2) * weight
        return self._filter_fix_weights(grad)

    def __call__(self, weights):
        return self.likelihood(weights)

    def likelihood(self, wt):
        l = self.f(wt)
        l = exp(l)
        return l

    def _fDummy(self, wt):
        # a dummy target function that is used when f is disabled
        if not hasattr(self, 'dummy_f'):
            self.dummyFCount = 0
        self.dummyFCount += 1
        if self.dummyFCount > 150:
            return 0
        print("self.dummyFCount", self.dummyFCount)

        if not hasattr(self, 'dummyFValue'):
            self.dummyFValue = 0
        if not hasattr(self, 'lastFullGradient'):
            self.dummyFValue = 0
        else:
            self.dummyFValue += sum(abs(self.lastFullGradient))
        print("_f: self.dummyFValue = ", self.dummyFValue)
        return self.dummyFValue

    def _filter_fix_weights(self, v):
        """
        Removes from the vector `v` all elements at indices that correspond to a fixed weight formula index.
        or a hard constraint formula.
        """
        if len(v) != len(self.mrf.formulas):
            raise Exception('Vector must have same length as formula weights')
        v_ = []  # numpy.zeros(len(v), numpy.float64)
        for val in [v[i] for i in range(len(self.mrf.formulas)) if
                    not self.mrf.mln.fix_weights[i] and self.mrf.mln.weights[i] != HARD]:
            v_.append(val)
        return v_

    def run(self, **params):
        """
        Learn the weights of the MLN given the training data previously
        loaded
        """
        if 'scipy' not in sys.modules:
            raise Exception("Scipy was not imported! Install numpy and scipy if you want to use weight learning.")
        # initial parameter vector: all zeros or weights from formulas
        self._w = [0] * len(self.mrf.formulas)
        for f in self.mrf.formulas:
            if self.mrf.mln.fix_weights[f.index] or self.use_init_weights or f.weight == HARD:
                self._w[f.index] = f.weight
        runs = 0
        print(self.use_init_weights, "weights is ")
        print(self._w)
        # pdb.set_trace()
        while runs < self.maxrepeat:
            self._prepare()
            self._optimize(**self._params)
            self._cleanup()
            runs += 1
            if not self.repeat():
                break
        return self.weights

    def _prepare(self):
        pass

    def _cleanup(self):
        pass

    def _optimize(self, optimizer='bfgs', **params):
        w = self._var_weights()
        if optimizer == "directDescent":
            opt = optimize.DirectDescent(w, self, **params)
        elif optimizer == "diagonalNewton":
            opt = optimize.DiagonalNewton(w, self, **params)
        else:
            opt = optimize.SciPyOpt(optimizer, w, self, **params)
        w = opt.run()
        # pdb.set_trace()
        self._w = self._add_fix_weights(w)

    def hessian(self, wt):
        wt = self._reconstructFullWeightVectorWithFixedWeights(wt)
        wt = list(map(float, wt))
        fullHessian = self._hessian(wt)
        return self._projectMatrixToNonFixedWeightIndices(fullHessian)

    def _projectMatrixToNonFixedWeightIndices(self, matrix):
        if len(self._fixedWeightFormulas) == 0:
            return matrix
        dim = len(self.mln.formulas) - len(self._fixedWeightFormulas)
        proj = numpy.zeros((dim, dim), numpy.float64)
        i2 = 0
        for i in range(len(self.mln.formulas)):
            if (i in self._fixedWeightFormulas):
                continue
            j2 = 0
            for j in range(len(self.mln.formulas)):
                if (j in self._fixedWeightFormulas):
                    continue
                proj[i2][j2] = matrix[i][j]
                j2 += 1
            i2 += 1
        return proj

    def _hessian(self, wt):
        raise Exception(
            "The learner '%s' does not provide a Hessian computation; use another optimizer!" % str(type(self)))

    def _f(self, wt, **params):
        raise Exception(
            "The learner '%s' does not provide an objective function computation; use another optimizer!" % str(
                type(self)))

    @property
    def name(self):
        if self.prior_stdev is None:
            sigma = 'no prior'
        else:
            sigma = "sigma=%f" % self.prior_stdev
        return "%s[%s]" % (self.__class__.__name__, sigma)


class DiscriminativeLearner(AbstractLearner):
    """
    Abstract superclass of all discriminative learning algorithms.
    Provides some convenience methods for determining the set of
    query predicates from the common parameters.
    """

    @property
    def qpreds(self):
        """
        Computes from the set parameters the list of query predicates
        for the discriminative learner. Eitehr the 'qpreds' or 'epreds'
        parameters must be given, both are lists of predicate names.
        """
        if not hasattr(self, '_preds'):
            qpreds = self._params.get('qpreds', [])
            if 'epreds' in self._params:
                epreds = self._params['epreds']
                qpreds.extend([p.name for p in self.mrf.predicates if p.name not in epreds])
                if not set(qpreds).isdisjoint(epreds):
                    raise Exception('Query predicates and evidence predicates must be disjoint.')
            if len(qpreds) == 0:
                raise Exception("For discriminative Learning, query or evidence predicates must be provided.")
            self._qpreds = qpreds
        return self._qpreds

    @property
    def epreds(self):
        return [p.name for p in self.mrf.predicates if p.name not in self.qpreds]

    def _qpred(self, predname):
        return predname in self.qpreds

    @property
    def name(self):
        return self.__class__.__name__ + "[query predicates: %s]" % ",".join(self.qpreds)


class SoftEvidenceLearner(AbstractLearner):
    """
    Superclass for all soft-evidence learners.
    """

    def __init__(self, mrf, **params):
        AbstractLearner.__init__(self, mrf, **params)

    def _getTruthDegreeGivenEvidence(self, gf, world=None):
        if world is None:
            world = self.mrf.evidence
        return gf.noisyor(world)
