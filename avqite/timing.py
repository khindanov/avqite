from mpi4py import MPI
from timeit import default_timer as timer


def timeit(method):
    def timed(*args, **kw):
        ts = timer()
        result = method(*args, **kw)
        te = timer()
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f' {method.__name__} {te - ts:.2f} s', flush=True)
        return result
    return timed
