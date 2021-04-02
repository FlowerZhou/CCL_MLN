import re
import time
import logging
import math
from math import exp, log
from math import fsum
import random
from dnutils import ifnone
import traceback
from functools import reduce
from collections import defaultdict


def merge_dom(*domains):
    """
    Returning a new domains dictionary that contains the elements of all the given domains
    """
    full_domain = {}
    for domain in domains:
        for domName, values in list(domain.items()):
            if domName not in full_domain:
                full_domain[domName] = set(values)
            else:
                full_domain[domName].update(values)
    for key, s in list(full_domain.items()):
        full_domain[key] = list(s)
    return full_domain


def logx(x):
    if x == 0:
        return - 100
    return math.log(x)


def tty(stream):
    isatty = getattr(stream, 'isatty', None)
    return isatty and isatty()


def cumsum(i, upto=None):
    return 0 if (not i or upto == 0) else reduce(int.__add__, i[:ifnone(upto, len(i))])


def rndbatches(i, size):
    i = list(i)
    random.shuffle(i)
    return batches(i, size)


def batches(i, size):
    batch = []
    for e in i:
        batch.append(e)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def balanced_parentheses(s):
    cnt = 0
    for c in s:
        if c == '(':
            cnt += 1
        elif c == ')':
            if cnt <= 0:
                return False
            cnt -= 1
    return cnt == 0


def get_index(list1, list2):
    """
    get the index of variable args to ground_atom args
    """
    index = []
    for lt in list2:
        for i in range(0, len(list1)):
            if list1[i] == lt:
                index.append(i)
    return index


def fstr(f):
    s = str(f)
    while s[0] == '(' and s[-1] == ')':
        s2 = s[1:-1]
        if not balanced_parentheses(s2):
            return s
        s = s2
    return s


def dict_union(d1, d2):
    """
    Returns a new dict containing all items from d1 and d2. Entries in d1 are
    overridden by the respective items in d2.
    """
    d_new = {}
    for key, value in d1.items():
        d_new[key] = value
    for key, value in d2.items():
        d_new[key] = value
    return d_new


def strip_comments(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " " # note: a space and not an empty string
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


def elapsed_time_str(elapsed):
    hours = int(elapsed / 3600)
    elapsed -= hours * 3600
    minutes = int(elapsed / 60)
    elapsed -= minutes * 60
    secs = int(elapsed)
    msecs = int((elapsed - secs) * 1000)
    return '{}:{:02d}:{:02d}.{:03d}'.format(hours, minutes, secs, msecs)


def headline(s):
    l = ''.ljust(len(s), '=')
    return '%s\n%s\n%s' % (l, s, l)


def parse_queries(mln, query_str):
    """
    Parses a list of comma-separated query strings.

    Admissible queries are all kinds of formulas or just predicate names.
    Returns a list of the queries.
    """
    queries = []
    query_preds = set()
    q = ''
    for s in map(str.strip, query_str.split(',')):
        if not s:
            continue
        if q != '': q += ','
        q += s
        if balanced_parentheses(q):
            try:
                # try to read it as a formula and update query predicates
                f = mln.logic.parse_formula(q)
                literals = f.literals()
                pred_names = [lit.pred_name for lit in literals]
                query_preds.update(pred_names)
            except:
                # not a formula, must be a pure predicate name
                query_preds.add(s)
            queries.append(q)
            q = ''
    if q != '':
        raise Exception('Unbalanced parentheses in queries: ' + q)
    return queries


def item(s):
    """
    Returns an arbitrary item from the given set `s`.
    """
    if not s:
        raise Exception('Argument of type {} is empty.'.format(type(s).__name__))
    for it in s:
        break
    return it


class CallbyRef(object):

    """
    Convenience class for treating any kind of variable as an object that can be
    manipulated in-place by a call-by-reference, in particular for primitive data types such as numbers.
    """
    def __init__(self, value):
        self.value = value


class edict(dict):

    def __add__(self, d):
        return dict_union(self, d)

    def __sub__(self, d):
        if type(d) in (dict, defaultdict):
            ret = dict(self)
            for k in d:
                del ret[k]
        else:
            ret = dict(self)
            del ret[d]
        return ret


INC = 1
EXC = 2


class Interval(object):

    def __init__(self, interval):
        tokens = re.findall(r'(\(|\[|\])([-+]?\d*\.\d+|\d+),([-+]?\d*\.\d+|\d+)(\)|\]|\[)', interval.strip())[0]
        if tokens[0] in ('(', ']'):
            self.left = EXC
        elif tokens[0] == '[':
            self.left = INC
        else:
            raise Exception('Illegal interval: {}'.format(interval))
        if tokens[3] in (')', '['):
            self.right = EXC
        elif tokens[3] == ']':
            self.right = INC
        else:
            raise Exception('Illegal interval: {}'.format(interval))
        self.start = float(tokens[1])
        self.end = float(tokens[2])

    def __contains__(self, x):
        return (self.start <= x if self.left == INC else self.start < x) and (
            self.end >= x if self.right == INC else self.end > x)


class temporary_evidence(object):
    """
    Context guard class for enabling convenient handling of temporary evidence in
    MRFs using the python `with` statement. This guarantees that the evidence
    is set back to the original whatever happens in the `with` block.

    :Example:

    >> with temporary_evidence(mrf, [0, 0, 0, 1, 0, None, None]) as mrf_:
    """

    def __init__(self, mrf, evidence=None):
        self.mrf = mrf
        self.evidence_backup = list(mrf.evidence)
        if evidence is not None:
            self.mrf.evidence = evidence

    def __enter__(self):
        return self.mrf

    def __exit__(self, exception_type, exception_value, tb):
        if exception_type is not None:
            traceback.print_exc()
            raise exception_type(exception_value)
        self.mrf.evidence = self.evidence_backup
        return True


class StopWatchTag:

    def __init__(self, label, start_time, stop_time=None):
        self.label = label
        self.start_time = start_time
        self.stop_time = stop_time

    @property
    def elapsed_time(self):
        return ifnone(self.stop_time, time.time()) - self.start_time

    @property
    def finished(self):
        return self.stop_time is not None


class StopWatch(object):
    """
    Simple tagging of time spans.
    """

    def __init__(self):
        self.tags = {}

    def tag(self, label, verbose=True):
        if verbose:
            print('%s...' % label)
        tag = self.tags.get(label)
        now = time.time()
        if tag is None:
            tag = StopWatchTag(label, now)
        else:
            tag.start_time = now
        self.tags[label] = tag

    def finish(self, label=None):
        now = time.time()
        if label is None:
            for _, tag in self.tags.items():
                tag.stop_time = ifnone(tag.stop_time, now)
        else:
            tag = self.tags.get(label)
            if tag is None:
                raise Exception('Unknown tag: %s' % label)
            tag.stop_time = now

    def __getitem__(self, key):
        return self.tags.get(key)

    def reset(self):
        self.tags = {}

    def print_steps(self):
        for t in sorted(self.tags.values(), key=lambda x: x.start_time):
            if t.finished:
                print('%s took %s' % (t.label, elapsed_time_str(t.elapsed_time)))
            else:
                print('%s is running for %s now...' % (t.label, elapsed_time_str(t.elapsed_time)))


# This function computes the probability of a triplet being true based on the MLN outputs.
def mln_triplet_prob(h, r, t, hrt2p):
    # KGE algorithms tend to predict triplets like (e, r, e), which are less likely in practice.
    # Therefore, we give a penalty to such triplets, which yields some improvement.
    if h == t:
        if hrt2p.get((h, r, t), 0) < 0.5:
            return -100
        return hrt2p[(h, r, t)]
    else:
        if (h, r, t) in hrt2p:
            return hrt2p[(h, r, t)]
        return 0.5


# This function reads the outputs from MLN and KGE to do evaluation.
# Here, the parameter weight controls the relative weights of both models.
def evaluate(mln_pred_file, kge_pred_file, output_file, weight):
    hit1 = 0
    hit3 = 0
    hit10 = 0
    mr = 0
    mrr = 0
    cn = 0

    hrt2p = dict()
    with open(mln_pred_file, 'r') as fi:
        for line in fi:
            h, r, t, p = line.strip().split('\t')[0:4]
            hrt2p[(h, r, t)] = float(p)

    with open(kge_pred_file, 'r') as fi:
        while True:
            truth = fi.readline()
            preds = fi.readline()

            if (not truth) or (not preds):
                break

            truth = truth.strip().split()
            preds = preds.strip().split()

            h, r, t, mode, original_ranking = truth[0:5]
            original_ranking = int(original_ranking)

            if mode == 'h':
                preds = [[pred.split(':')[0], float(pred.split(':')[1])] for pred in preds]

                for k in range(len(preds)):
                    e = preds[k][0]
                    preds[k][1] += mln_triplet_prob(e, r, t, hrt2p) * weight

                preds = sorted(preds, key=lambda x:x[1], reverse=True)
                ranking = -1
                for k in range(len(preds)):
                    e = preds[k][0]
                    if e == h:
                        ranking = k + 1
                        break
                if ranking == -1:
                    ranking = original_ranking

            if mode == 't':
                preds = [[pred.split(':')[0], float(pred.split(':')[1])] for pred in preds]

                for k in range(len(preds)):
                    e = preds[k][0]
                    preds[k][1] += mln_triplet_prob(h, r, e, hrt2p) * weight

                preds = sorted(preds, key=lambda x:x[1], reverse=True)
                ranking = -1
                for k in range(len(preds)):
                    e = preds[k][0]
                    if e == t:
                        ranking = k + 1
                        break
                if ranking == -1:
                    ranking = original_ranking

            if ranking <= 1:
                hit1 += 1
            if ranking <=3:
                hit3 += 1
            if ranking <= 10:
                hit10 += 1
            mr += ranking
            mrr += 1.0 / ranking
            cn += 1

    mr /= cn
    mrr /= cn
    hit1 /= cn
    hit3 /= cn
    hit10 /= cn

    print('MR: ', mr)
    print('MRR: ', mrr)
    print('Hit@1: ', hit1)
    print('Hit@3: ', hit3)
    print('Hit@10: ', hit10)

    with open(output_file, 'w') as fo:
        fo.write('MR: {}\n'.format(mr))
        fo.write('MRR: {}\n'.format(mrr))
        fo.write('Hit@1: {}\n'.format(hit1))
        fo.write('Hit@3: {}\n'.format(hit3))
        fo.write('Hit@10: {}\n'.format(hit10))


def augment_triplet(pred_file, trip_file, out_file, threshold):
    with open(pred_file, 'r') as fi:
        data = []
        for line in fi:
            l = line.strip().split()
            data += [(l[0], l[1], l[2], float(l[3]))]

    with open(trip_file, 'r') as fi:
        trip = set()
        for line in fi:
            l = line.strip().split()
            trip.add((l[0], l[1], l[2]))

        for tp in data:
            if tp[3] < threshold:
                continue
            trip.add((tp[0], tp[1], tp[2]))

    with open(out_file, 'w') as fo:
        for h, r, t in trip:
            fo.write('{}\t{}\t{}\n'.format(h, r, t))
