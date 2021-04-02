import multiprocessing
from multiprocessing import pool
import traceback
import sys
import signal
import os
import psutil
from functools import wraps


class CtrlException(Exception):
    pass


class with_tracing(object):
    """
    Wrapper class for functions intended to be executed in parallel
    on multiple cores. This facilitates debugging with multiprocessing.
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        signal.signal(signal.SIGINT, signal_handler)
        try:
            result = self.func(*args, **kwargs)
            return result
        except CtrlException:
            pass
        except Exception as e:
            traceback.print_exc()
            raise e


def signal_handler(signal, frame):
    sys.stderr.write('Terminating process %s.\n' % os.getpid())
    raise KeyboardInterrupt()
    sys.exit(0)


class _methodcaller:
    """
    Convenience class for calling a method of an object in a worker pool
    """

    def __init__(self, method, sideeffects=False):
        self.method = method
        self.sideeffects = sideeffects

    def __call__(self, args):
        check_mem()
        if type(args) is list or type(args) is tuple:
            inst = args[0]
            args = args[1:]
        else:
            inst = args
            args = []
        if self.sideeffects:
            ret = getattr(inst, self.method)(*args)
            return ret, inst.__dict__
        return getattr(inst, self.method)(*args)


def check_mem():
    if float(psutil.virtual_memory().percent) > 95.:
        raise Exception('Aborting due to excessive memory consumption.')


def make_mem_safe():
    if sys.platform.startswith('linux'):
        import resource
        import psutil
        for rsrc in (resource.RLIMIT_AS, resource.RLIMIT_DATA):
            freemem = psutil.virtual_memory().free
            hard = int(round(freemem * .8))
            soft = hard
            resource.setrlimit(rsrc, (soft, hard))  # limit to 80% of available memory


class NoDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class NonDaemonicPool(pool.Pool):
    Process = NoDaemonProcess


# example how to be used
if __name__ == '__main__':
    def f(x):
        return x * x
    pool = multiprocessing.Pool(processes=4)  # start 4 worker processes
    print(pool.map(with_tracing(f), list(range(10))))  # prints "[0, 1, 4,..., 81]"
