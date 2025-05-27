# author: Yongxin Yao (yxphysice@gmail.com)
from mpi4py import MPI
import json, numpy


class model:
    '''
    Generates a model object given an incar file.

    Attributes
    ----------
    _mrank : int
        Rank of an MPI process
    _filename : str
        Filename of the incar file.
    _localh : bool
        Flag whether to use local_exect for h.
        If False, amplitude is used.
    _incar : str
        Incar file content.
    _nsite : int
        Number of sites in the problem.
    _h : List[str]
        List of Pauli strings (with corresponding coeffs)
        in the Hamiltonian.
    Generates a model object given an incar file.
    '''
    def __init__(self,
            localh=False,
            filename = "1",
            model_dir = "1/"
            ):
        comm = MPI.COMM_WORLD
        self._mrank = comm.Get_rank()
        self._filename = filename
        self._model_dir = model_dir

        self.load_incar()
        self.set_h()
        self._localh = localh

    def load_incar(self):
        self._incar = json.load(open(self._model_dir + "incars/incar" + self._filename, "r"))

    def set_h(self):
        hs_list = self._incar["h"]
        if self._mrank == 0:
            print(f' Hamiltonian terms: {len(hs_list)}', flush=True)
        self._nsite = len(hs_list[0].split("*")[1])

        self._h = hs_list
