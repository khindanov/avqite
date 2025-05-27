# authors: CyQC group (yxphysice@gmail.com)
import itertools
import os
import pickle
import time
import warnings

import cotengra as ctg
import numpy
import quimb.tensor as qtn
import scipy.linalg
import scipy.optimize
import scipy.sparse
from mpi4py import MPI
from qiskit.quantum_info import SparsePauliOp

from . import quimbify
from .timing import timeit

# removing parallel=False seems to speed up the optimization
opt_eco1 = ctg.ReusableHyperOptimizer(
    # parallel=False,
    # just do a few runs
    max_repeats=32,
    # only use the basic greedy optimizer ...
    methods=["greedy"],
    # ... but pair it with reconfiguration
    reconf_opts={},
    # use dynamic slicing to target a width of 30
    slicing_reconf_opts={"target_size": 2**28},
    # just uniformly sample the space
    optlib="random",
    # terminate search if contraction is cheap
    max_time="rate:1e9",
    # account for both flops and write - usually wise for practical performance
    minimize="combo",
)


class ansatz:
    """
    define the main procedures of one step of avqite calculation.
    set up: 1) default reference state of the ansatz.
            2) operator pool

    Attributes
    ----------
    _ansatz_pids : List[ind]
        Indices of the ansatz operators within the pool.

    """

    def __init__(
        self,
        model,  # the qubit model
        rcut=1.0e-2,  # McLachlan distance cut-off
        fcut=1.0e-1,  # cut-off ratio for invidual contribution
        max_add=5,  # maximal number of new pauli rotation gates to be added
        maxntheta=-1,  # maximal variational parameters allowed
        bounds=(-5, 5),  # bounds for dtheta/dt
        delta=1.0e-4,  # Tikhonov regularization parameter if >= 0,
        # otherwise switch to lsq_linear
        invmode=0,  # linear equation solver mode.
        dt=0.1,  # time step
        tsave=1200,  # time interval to save intermediate results
        vtol=1e-4,  # gradient cutoff
        tetras=True,  # use tetras by default
        tmax=1e7,  # maximal time to simulate
        tf=numpy.inf,  # maximal final (algorithmic) time it can reach
        filename="1",
        model_dir="1/",
        optimize="greedy",
        simplify_sequence="ADCRS",
        backend="None",
    ):
        self._comm = MPI.COMM_WORLD
        self._mrank = self._comm.Get_rank()
        self._msize = self._comm.Get_size()

        self._model = model
        self._nq = model._nsite
        self._state = None
        self._rcut = rcut
        self._rcut0 = rcut * fcut
        self._max_add = max_add
        self._maxntheta = maxntheta
        self._bounds = bounds
        self._delta = delta
        self._invmode = invmode
        self._dt = dt
        self._dthdt = None
        self._tsave = tsave
        self._vtol = vtol
        self._tetras = tetras
        self._tmax = tmax
        self._tf = tf
        self._filename = filename
        self._model_dir = model_dir
        self._optimize = optimize
        self._simplify_sequence = simplify_sequence
        if backend == "None":
            self._backend = None
        else:
            self._backend = backend
        self.max_tn_width = self.max_tn_cost = -1

        # generate operator pool
        self._mmat = None
        self._mzeros = []
        self.set_config()
        self.setup_pool()
        self.set_pstrings_hp()
        self.init_ansatz()
        self.init_mcircs()

        # create folders to store outs and data
        self.outputs_dir = os.path.join(self._model_dir, "outputs")
        self.outputs_dir_model = os.path.join(
            self._model_dir, "outputs", self._filename
        )
        self.data_dir = os.path.join(self._model_dir, "data")
        self.data_dir_model = os.path.join(self._model_dir, "data", self._filename)
        os.makedirs(self.outputs_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.outputs_dir_model, exist_ok=True)
        os.makedirs(self.data_dir_model, exist_ok=True)

    def init_ansatz(self):
        self._ansatz = []
        # set reference state
        self.set_ref_circ()
        # variational parameters.
        self._params = []
        # the op indices in pool for the ansatz
        self._ansatz_pids = []
        self._ngates = [0] * self._nq
        self._t = 0
        self._layer_range = []
        self._iter = 0
        self._base_circs = []

        # ansatz_simp_filename = (self.script_dir + f"data/{self._filename}/ansatz_simp_tetras{self._tetras}.pkle")
        # if os.path.isfile(ansatz_simp_filename):
        #     self.pload_ansatz_simp(ansatz_simp_filename)
        # ansatz_inp_filename = (self.script_dir + f"data/{self._filename}/ansatz_inp_o"+self._config["optimize"]+"_s"+
        #               self._config["simplify_sequence"]+"_b"+str(self._config["backend"])+f"_iter{self._iter}_tetras{self._tetras}.pkle")
        # if os.path.isfile(ansatz_inp_filename):
        #     self.pload_ansatz_inp(ansatz_inp_filename)

        self.set_base_circs()

        if self._mrank == 0:
            print(f" nparams in the initial ansatz: {len(self._ansatz_pids)}")

    def set_config(self):
        self._config = {
            "optimize": self._optimize,
            "simplify_sequence": self._simplify_sequence,
            "backend": self._backend,
        }

    @property
    def ngates(self):
        return self._ngates[:]

    @timeit
    def setup_pool(self):
        """
        setup pool from incar.
        """
        raise NotImplementedError

    def set_ref_circ(self):
        """set reference circ."""
        self._ref_circ = qtn.Circuit(self._nq)
        # convension flip compared with previous avqite implementation due to binary interpretation.
        ref_state = self._model._incar["ref_state"]
        assert ref_state.count("0") + ref_state.count("1") == len(ref_state)
        for i, el in enumerate(ref_state):
            if el == "1":
                self._ref_circ.apply_gate("X", i)

    def update_ngates(self):
        """
        update gate counts.
        """
        raise NotImplementedError

    def get_dthdt(self, mmat, vvec, rtol=1e-6):
        if self._delta < 0:
            # slow for large dimension
            res = scipy.optimize.lsq_linear(
                mmat,
                vvec,
                bounds=self._bounds,
                # lsq_solver='lsmr',
            )
            dthdt = res["x"]
        else:
            a = mmat + self._delta * numpy.eye(mmat.shape[0])
            if self._invmode == 0:
                ainv = numpy.linalg.inv(a)
                dthdt = ainv.dot(vvec)
            else:
                # use cg.
                if self._dthdt is None:
                    x0 = None
                elif len(self._dthdt) == len(vvec):
                    x0 = self._dthdt
                else:
                    x0 = numpy.zeros_like(vvec)
                    x0[: len(self._dthdt)] = self._dthdt

                dthdt, info = scipy.sparse.linalg.cg(a, vvec, x0=x0, rtol=rtol)
                assert info == 0, f"info = {info} from scipy.sparse.linalg.cg."
        return dthdt

    def get_scores(self, inds_chk):
        scores = numpy.zeros(max(inds_chk) + 1)
        self._mmat_pp_buf *= 0
        for i, ind in enumerate(inds_chk):
            if i % self._msize == self._mrank:
                scores[ind], _, _, _ = self.get_score([ind])

        scores_allreduce = numpy.empty(numpy.shape(scores), dtype=float)
        self._comm.Allreduce([scores, MPI.DOUBLE], [scores_allreduce, MPI.DOUBLE])
        scores = scores_allreduce

        mmat_pp_buf_allreduce = numpy.empty(numpy.shape(self._mmat_pp_buf), dtype=float)
        self._comm.Allreduce(
            [self._mmat_pp_buf, MPI.DOUBLE], [mmat_pp_buf_allreduce, MPI.DOUBLE]
        )
        self._mmat_pp_buf = mmat_pp_buf_allreduce

        where = numpy.where(abs(self._mmat_pp_buf) > 1e-17)[0]
        self._mmat_pp[where] = self._mmat_pp_buf[where]
        scores_shft = -numpy.ones(max(inds_chk) + 1)
        scores_shft[numpy.r_[inds_chk]] = self._scores_bias[numpy.r_[inds_chk]]
        scores += scores_shft
        return numpy.asarray(scores)

    def get_score(self, inds, parallel=False):
        mmat, vvec = self.get_mv(inds, parallel=parallel)
        dthdt = self.get_dthdt(mmat, vvec)
        dist_p = vvec.dot(dthdt)
        score = dist_p.real - self._distp
        return score, mmat, vvec, dthdt

    @timeit
    def add_ops(self):
        """
        ansatz adaptively expanding procedure in avqite by adding
        layer by layer.
        """
        self._ntheta_old = len(self._params)
        npool = len(self._pool)
        if self._maxntheta > 0 and self._ntheta_old >= self._maxntheta:
            return

        icyc = 0
        # energy variance
        hvar = self._e2 - self._e**2
        self.eval_vpool()
        self.eval_mmat_pt()
        self._inds_add = []

        # now layer
        inds_chk = list(range(npool))

        for _ in range(self._max_add):
            scores = self.get_scores(inds_chk)
            isort = scores.argsort()[::-1]
            ichoose = []
            nlayer = len(self._layer_range)
            qsupport = []
            if nlayer == 0:
                self._layer_range.append([0, 0])
            else:
                imin = self._layer_range[-1][1]
                self._layer_range.append([imin, imin])

            for i, idx in enumerate(isort):
                if scores[idx] < 0:
                    break

                if idx in ichoose or numpy.any(
                    [self._pool[idx][iq] != "I" for iq in qsupport]
                ):
                    continue

                ichoose.append(idx)
                inds_chk.remove(idx)

                self._ansatz_pids.append(idx)
                self._ansatz.append(self._pool[idx])
                self.update_ngates()
                self._params.append(0)
                self._layer_range[-1][1] += 1

                for iq, s in enumerate(self._pool[idx]):
                    if s != "I":
                        qsupport.append(iq)
                if len(qsupport) == self._nq or not self._tetras:
                    break

            # update mmat, vvec
            if len(ichoose) == 0:
                warnings.warn("no more unitaries can be added.")
                break

            diff, mmat, vvec, dthdt = self.get_score(ichoose, parallel=True)
            # just in case any difference for dthdt
            dthdt = self._comm.bcast(dthdt, root=0)
            self._comm.Barrier()

            dist_p = diff + self._distp
            dist = hvar - dist_p
            self._distp = dist_p
            self._mmat = mmat
            self._vvec = vvec
            self._dthdt = dthdt
            self._inds_add.extend(ichoose)

            if self._mrank == 0:
                if self._nq <= 12:
                    print(
                        " add op:",
                        [
                            self._ansatz[i]
                            for i in range(
                                self._layer_range[-1][0], self._layer_range[-1][1]
                            )
                        ],
                    )
                print(
                    " add op:",
                    [
                        int(self._ansatz_pids[i])
                        for i in range(
                            self._layer_range[-1][0], self._layer_range[-1][1]
                        )
                    ],
                )
                print(
                    " grad:",
                    [
                        f"{vvec[i]:.6f}"
                        for i in range(
                            self._layer_range[-1][0], self._layer_range[-1][1]
                        )
                    ],
                )
                print(f"icyc = {icyc}, dist = {dist:.2e}, improving {diff:.2e}")
            icyc += 1

            if self._maxntheta > 0 and mmat.shape[0] >= self._maxntheta:
                break
            if dist < self._rcut:
                break
        if self._mrank == 0:
            print(f" get_mv circs done: {len(numpy.where(self._mmat_pp < 0.5)[0])}")
            print(f" number of layers in the ansatz: {len(self._layer_range)}")
            print(f" final mcLachlan distance: {dist:.2e}")

    def get_mv(self):
        raise NotImplementedError

    @timeit
    def run(self):
        # keep time to save intermediate results
        if self._mrank == 0:
            twall0 = time.time()

        while True:
            self.one_step()
            var_h = self._e2 - self._e**2
            if self._mrank == 0:
                print(f" ngates: {self._ngates}")
                print(
                    f"iter = {self._iter}, t = {self._t:.4f}, e = {self._e:.8f}, "
                    + f"vmax = {self._vmax:.2e} var_h = {var_h:.2e}\n",
                    flush=True,
                )

            if self._vmax < self._vtol or abs(self._tf - self._t) < 1e-7:
                break

            if self._mrank == 0:
                twall1 = time.time()
                if twall1 - twall0 > self._tsave:
                    self.psave_ansatz_simp()
                    twall0 = twall1

            # if self._mrank == 0:
            #     if self._iter % 10 == 0:
            #         self.psave_ansatz_simp()

            self._iter += 1
            if self._t >= self._tmax:
                break

        if self._mrank == 0:
            print(f"total layers in ansatz: {len(self._layer_range)}")

    @timeit
    def one_step(self):
        self._mmat = self.calc_mmat()
        self.update_ansatz_circ_params()
        self._e = e = self.calc_e()
        self._e2 = e2 = self.calc_e2()

        # self._vvec = self.calc_v_ps()
        self._vvec = self.calc_v_pd()

        if len(self._vvec) > 0:
            if self._mrank == 0:
                dthdt = self.get_dthdt(self._mmat, self._vvec)
            else:
                dthdt = None
            dthdt = self._comm.bcast(dthdt, root=0)
            self._comm.Barrier()
        else:
            dthdt = numpy.asarray([])
        # McLachlan distance
        dist_p = self._vvec.dot(dthdt)
        dist_p = dist_p.real
        dist_h2 = e2 - e**2
        dist = dist_h2 - dist_p

        self._distp = dist_p
        if len(dthdt) > 0:
            pthmax = numpy.max(numpy.abs(dthdt))
        else:
            pthmax = 0
        self._dthdt = dthdt
        if self._mrank == 0:
            print(f" initial mcLachlan distance: {dist:.2e} pthmax: {pthmax:.2f}")
        if dist > self._rcut or len(self._params) == 0:
            self.add_ops()
            self.set_base_circs()
            self.set_mcircs(self._ntheta_old)
            if len(self._dthdt) > 0:
                pthmax = numpy.max(numpy.abs(self._dthdt))
            else:
                pthmax = 0
            if self._mrank == 0:
                print(f" pthmax: {pthmax:.2f}")

        self.evolve_theta()
        self.reset_mcircs()

        # maximal gradient element
        self._vmax = numpy.max(numpy.abs(self._vvec))

    def evolve_theta(self):
        # up to self._tf if needed.
        dt = min(self._dt, self._tf - self._t)
        if len(self._dthdt) > 0:
            pthmax = numpy.max(numpy.abs(self._dthdt))
            if pthmax > 0:
                # similar step size constraints for dtheta_max
                dt = min(dt, dt / pthmax, self._tmax - self._t)
            self._params = [p + pp * dt for p, pp in zip(self._params, self._dthdt)]
        self._t += dt

    def psave_ansatz_simp(self):
        # pass
        # for continuous run
        with open(
            self.script_dir
            + f"data/{self._filename}/ansatz_simp_o"
            + self._config["optimize"]
            + "_s"
            + self._config["simplify_sequence"]
            + "_b"
            + str(self._config["backend"])
            + f"_iter{self._iter}_tetras{self._tetras}.pkle",
            "wb",
        ) as f:
            data = [
                self._ansatz_pids,
                self._params,
                self._ngates,
                self._layer_range,
                self._t,
                self._iter,
            ]
            pickle.dump(data, f)

    def psave_ansatz_inp(self):
        # should be transferable.
        with open(
            self.script_dir
            + f"data/{self._filename}/ansatz_inp_o"
            + self._config["optimize"]
            + "_s"
            + self._config["simplify_sequence"]
            + "_b"
            + str(self._config["backend"])
            + f"_iter{self._iter}_tetras{self._tetras}.pkle",
            "wb",
        ) as f:
            data = [
                self._ansatz,
                self._params,
            ]
            pickle.dump(data, f)

    def psave_tn_w_c(self, width, cost, tn_type, freq=10):
        pass
        # if self._iter % freq == 0:
        #     with open(self.script_dir + f"data-width-cost/{self._filename}notetras/data_wc_calc_{tn_type}_"+
        #               f"iter{self._iter}_o"+self._config["optimize"]+"_s"+
        #               self._config["simplify_sequence"]+"_b"+str(self._config["backend"])+f"_tetras{self._tetras}.pkle", "wb") as f:
        #         data = [width, cost]
        #         pickle.dump(data, f)

    def psave_circ(self, circ, tn_type, width, cost):
        pass
        # ranks = [0,10,20,30,40,50,60]
        # if self._mrank in ranks:
        #     with open(self.script_dir + f"data-circuits/{self._filename}notetras/data_circ_rank{self._mrank}_tetras{self._tetras}.pkle", "wb") as f:
        #         data=[width,
        #             cost,
        #             circ.gates,
        #             tn_type,
        #             self._iter,
        #             self._config["optimize"],
        #             self._config["simplify_sequence"],
        #             str(self._config["backend"])
        #             ]
        #         pickle.dump(data, f)

    def pload_ansatz_simp(self, ansatz_filename):
        if self._mrank == 0:
            with open(ansatz_filename, "rb") as f:
                data = pickle.load(f)
            data = MPI.pickle.dumps(data)
        else:
            data = None
        data = self._comm.bcast(data, root=0)
        self._comm.Barrier()

        [
            self._ansatz_pids,
            self._params,
            self._ngates,
            self._layer_range,
            self._t,
            self._iter,
        ] = MPI.pickle.loads(data)

        self.set_ansatz_from_pids()

    def set_ansatz_from_pids(self):
        for idx in self._ansatz_pids:
            self._ansatz.append(self._pool[idx])

    def pload_ansatz_inp(self, ansatz_filename):
        if self._mrank == 0:
            with open(ansatz_filename, "rb") as f:
                data = pickle.load(f)
            data = MPI.pickle.dumps(data)
        else:
            data = None
        data = self._comm.bcast(data, root=0)
        self._comm.Barrier()
        [self._ansatz, self._params] = MPI.pickle.loads(data)

        self._ansatz_pids = []
        labels = self._pool
        self._layer_range.append([0, 0])
        qsupport = []

        for i, label in enumerate(self._ansatz):
            idx = labels.index(label)
            self._ansatz_pids.append(idx)
            self.update_ngates(i)
            for iq, s in enumerate(label):
                if s != "I":
                    if iq in qsupport:
                        self._layer_range[-1][1] = i
                        self._layer_range.append([i, 0])
                        qsupport = []
                        for iqp, sp in enumerate(label[:iq]):
                            if sp != "I":
                                qsupport.append(iqp)
                    qsupport.append(iq)
        self._layer_range[-1][1] = len(self._ansatz)


class ansatzSinglePool(ansatz):
    """
    adaptvqite with single operator pool.

    Attributes
    ----------
    _config : Dict[str : str]
        Configuration file with Quimb parameters.
    _pool : List[str]
        List of Pauli strings from the pool.
    _mrank : int
        Rank of an MPI process.
    _scores_bias : numpy.ndarray

    _poolpgates : List[quimb.tensor.circuit.Gate]
        List of Quimb gates corresponding to the operators in the pool.
    _hcoeffs : numpy.ndarray
        Coefficients for each Pauli string in the Hamiltonian (h).
    _pstrings_h : List[str]
        List of Pauli strings in the Hamiltonian turned into Quimb gates.
    _hsize : int
        Number of Pauli strings in the Hamiltonian.
    _hop : SparsePauliOp
        Hamiltonian in the form of Qiskit SparsePauliOp.
    _hhcoeffs : numpy.ndarray
        Coefficients for each Pauli string in h^2.
    _pstrings_hh : List[str]
        List of Pauli strings in h^2 (deleted at the end).
        _pstrings_hh excludes Pauli strings from h.
    _hhsize : int
        Number of Pauli strings in _pstrings_hh.
    _hhinds : numpy.ndarray
        Should be related to the indices h^2 Pauli strings within the combined
        sorted (h,h^2) list, but not exactly sure
    _pstrings_hhlocal : List[str]
        List of Pauli strings in h^2 scattered over MPI processes
        turned into Quimb gates.
    _pstrings_ph : List[str]
        List of Pauli strings in pool*h (deleted at the end).
        _pstrings_ph excludes Pauli strings from h and h^2.
    _phsize : int
        Number of Pauli strings in _pstrings_ph.
    _poolh : List[List]
        _poolh[0][i] is a list of coefficients of Paulis in the product of i'th
        Pool operator and the Paulis from h.
        _poolh[1][i] should be related to the indices pool[i]*h Pauli strings
        within the combined sorted (h,h^2,_pstrings_ph) list, but not exactly sure
    _pstrings_phlocal : List[str]
        List of Pauli strings in pool*h scattered over MPI processes
        turned into Quimb gates.
    _mcircs : List[quimb.tensor.circuit]
        List of circuits needed to compute the entire matrix M_{ij}
    _minds : List[]
        List of indices M_{ij} to which _mcircs correpond.
        _minds[0] is the list of indices i, _minds[1] is the list of indices j.

    """

    @timeit
    def setup_pool(self, tol=1e-8):
        """
        setup pool from incar.
        """
        labels = self._model._incar["pool"]
        self._pool = labels
        if self._mrank == 0:
            print(f" pool dimension: {len(self._pool)}")

        # setup score bias
        # effectively inds are indices of the pool sorted alphabetically
        # not sure why numpy.argsort(labels) is not enough here
        # _scores_bias is some small number which is different for each index
        inds = numpy.lexsort((labels, labels))
        self._scores_bias = inds * tol / len(labels)
        self.set_pool_gates()

    def get_poolrgate(self, i):
        """
        Returns Quimb gates corresponding to the Pauli rotation gate with
        i'th Pauli string from the pool and angle=0.
        """
        circ = quimbify.add_pauli_rotation_gate(
            qc=qtn.Circuit(N=self._nq),
            pauli_string=self._pool[i],
            decompose_rzz=False,
        )
        return circ.gates

    def set_pool_gates(self):
        # not good, the params will be shared among different occurences.
        # pauli string roataion gates
        # self._poolrgates = [
        #         quimbify.add_pauli_rotation_gate(
        #             qc=qtn.Circuit(N=self._nq),
        #             pauli_string=p,
        #             decompose_rzz=False,
        #             ).gates
        #             for p in self._pool
        #         ]
        # pauli string gate
        self._poolpgates = [quimbify.pauli_string_to_quimb_gates(p) for p in self._pool]

    def set_base_circs(self):
        """
        Sets up base circuits that represent the ansatz state and all
        intermediate states.
        Last element in the list _base_circs is the ansatz state.
        Can be used adaptively to expand _base_circs as the ansatz expands.
        """
        ncircs = len(self._base_circs)
        if ncircs == 0:
            circ = self._ref_circ
        else:
            circ = self._base_circs[ncircs - 1]

        for ind in self._ansatz_pids[ncircs:]:
            circ = circ.copy()
            circ.apply_gates(
                self.get_poolrgate(ind),
                contract=False,
            )
            self._base_circs.append(circ)
        assert len(self._base_circs) == len(self._params)

    def init_mcircs(self):
        self._minds = [[], []]
        self._mcircs = []
        self._nmcircs = 0
        self.set_mcircs(0)

    def set_mcircs(self, nth0):
        nth = len(self._params)
        nij = (nth - 1) * nth // 2
        if nth <= 1 or nth0 == nth:
            return

        if nth0 > 0:
            icount = self._nmcircs
        else:
            icount = 0

        zeros = numpy.zeros(nij, dtype=bool)
        zeros[self._mzeros] = True

        # we need to only generate circuits that haven't been generated already
        for i in range(nth0, nth):
            for j in range(i):
                ij = i * (i - 1) // 2 + j
                if zeros[ij]:
                    continue
                # parallelizing different elements of M matrix
                if icount % self._msize == self._mrank:
                    self._minds[0].append(i)
                    self._minds[1].append(j)
                    self._mcircs.append(self.get_mcirc(i, j))
                icount += 1

        if self._mrank == 0:
            print(f" total nontrivial mcircs: {icount}")
        self._nmcircs = icount

    def reset_mcircs(self):
        # get lower triangular part of mmat
        np = self._mmat.shape[0]
        mpart = self._mmat[numpy.tril_indices(np, k=-1)]

        wheres = numpy.where(abs(mpart) < 1e-12)[0]
        if self._mrank == 0:
            print(f" Zero mmat elements: {len(wheres)}/{len(mpart)}")

        n0 = len(self._mzeros)
        if n0 == len(wheres) and numpy.allclose(self._mzeros, wheres):
            if n0 > (np - 1) * np // 2 - self._nmcircs + 10:
                self.init_mcircs()
        else:
            self._mzeros = wheres

    def get_mcirc(self, i, j, gate_i=None, gate_j=None):
        """
        Sets up a circuit for computing M_{ji} through amplitude evaluation.
        """
        assert i > j
        if gate_i is None:
            gate_i = self._poolpgates[self._ansatz_pids[i]]
        if gate_j is None:
            gate_j = self._poolpgates[self._ansatz_pids[j]]

        if j < 0:
            circ = self._ref_circ.copy()
        else:
            circ = self._base_circs[j].copy()
        circ.apply_gates(
            gate_j,
            contract=False,
        )

        for k in range(j + 1, i):
            circ.apply_gates(
                self.get_poolrgate(self._ansatz_pids[k]),
                contract=False,
            )
        circ.apply_gates(
            gate_i,
            contract=False,
        )

        # to be set with -thetas
        for k in reversed(range(i)):
            circ.apply_gates(
                self.get_poolrgate(self._ansatz_pids[k]),
                contract=False,
            )
        params_dict = circ.get_params()
        assert len(params_dict) == 2 * i
        return circ

    def calc_mmat_i(self, i, circ):
        params_dict = circ.get_params()
        for j, key in enumerate(params_dict):
            if j < i:
                params_dict[key] = [self._params[j]]
            else:
                params_dict[key] = [-self._params[2 * i - j - 1]]

        circ.set_params(params_dict)
        reh = circ.amplitude_rehearse(
            self._model._incar["ref_state"],
            optimize=self._config["optimize"],
            simplify_sequence=self._config["simplify_sequence"],
        )

        # if reh['W'] > 25:
        #     reh = circ.amplitude_rehearse(
        #             self._model._incar["ref_state"],
        #             optimize=opt_eco1,
        #             simplify_sequence=self._config["simplify_sequence"],
        #             )

        res = reh["tn"].contract(
            all,
            optimize=reh["tree"],
            output_inds=(),
            backend=self._config["backend"],
        )
        res = numpy.real(res) / 4

        if reh["W"] > self.max_tn_width:
            self.psave_circ(circ, "mmat", reh["W"], reh["C"])
            self.max_tn_width = reh["W"]
            self.max_tn_cost = reh["C"]

        return res, reh["W"], reh["C"]

    @timeit
    def calc_mmat(self):
        nth = len(self._params)
        mmat = numpy.zeros((nth, nth))
        mmat_width = numpy.zeros((nth, nth))
        mmat_cost = numpy.zeros((nth, nth))
        mvals = []
        mvals_width = []
        mvals_cost = []
        assert len(self._mcircs) == len(self._minds[0])

        if self._mrank == 0:
            print(f" calc_mmat circs to run: {self._nmcircs}")

        for ic, circ, i in zip(itertools.count(), self._mcircs, self._minds[0]):
            res, width, cost = self.calc_mmat_i(i, circ)
            mvals.append(res)
            mvals_width.append(width)
            mvals_cost.append(cost)

        mmat[self._minds[0], self._minds[1]] = mmat[self._minds[1], self._minds[0]] = (
            mvals
        )
        mmat_width[self._minds[0], self._minds[1]] = mmat_width[
            self._minds[1], self._minds[0]
        ] = mvals_width
        mmat_cost[self._minds[0], self._minds[1]] = mmat_cost[
            self._minds[1], self._minds[0]
        ] = mvals_cost

        mmat_allreduce = numpy.empty(numpy.shape(mmat), dtype=float)
        self._comm.Allreduce([mmat, MPI.DOUBLE], [mmat_allreduce, MPI.DOUBLE])
        mmat = mmat_allreduce
        mmat += numpy.eye(nth) / 4

        mmat_width_reduce = numpy.empty(numpy.shape(mmat_width), dtype=float)
        self._comm.Reduce(
            [mmat_width, MPI.DOUBLE], [mmat_width_reduce, MPI.DOUBLE], root=0
        )

        mmat_cost_reduce = numpy.empty(numpy.shape(mmat_cost), dtype=float)
        self._comm.Reduce(
            [mmat_cost, MPI.DOUBLE], [mmat_cost_reduce, MPI.DOUBLE], root=0
        )

        if self._mrank == 0:
            nzeros = numpy.count_nonzero(abs(mmat) < 1e-12)
            print(
                f" zero fraction of mmat: {nzeros}/{mmat.size} or {nzeros // 2}/{(mmat.size - len(mmat)) // 2}"
            )

            self.psave_tn_w_c(mmat_width_reduce, mmat_cost_reduce, "mmat")

        return mmat

    @timeit
    def calc_v_ps(self):
        np = len(self._params)
        vvec = numpy.zeros(np)
        if np == 0:
            return vvec

        vals = numpy.zeros((np, 2, self._hsize))
        if self._model._localh:
            params_dict = self._base_circs[-1].get_params()
            circ = self._base_circs[-1].copy()
            for i, key in enumerate(params_dict):
                params_dict[key] = [self._params[i]]
        else:
            params = self._params.copy()

        if self._mrank == 0:
            print(f" calc_v_ps circs to run: {np * 2 * self._hsize}")

        icount = 0
        for i, x in zip(itertools.count(), self._params):
            if self._model._localh:
                key = list(params_dict.keys())[i]
            for j, dtheta in enumerate([numpy.pi / 2, -numpy.pi / 2]):
                if self._model._localh:
                    params_dict[key] = [x + dtheta]
                    circ.set_params(params_dict)
                else:
                    params[i] = x + dtheta
                for k, op in enumerate(self._pstrings_h):
                    if icount % self._msize == self._mrank:
                        if self._model._localh:
                            exp_val = circ.local_expectation(
                                op[0],
                                op[1],
                                optimize=self._config["optimize"],
                                simplify_sequence=self._config["simplify_sequence"],
                            )
                        else:
                            exp_val = self.get_expectation_via_amplitude(
                                op, params=params
                            )[0]
                        assert abs(exp_val.imag) < 1e-7
                        vals[i, j, k] = exp_val.real
                    icount += 1
                if self._model._localh:
                    params_dict[key] = [x]
                else:
                    params[i] = x

        vals = self._comm.allreduce(vals)

        vals = numpy.einsum(
            "ijk,k->ij",
            vals,
            self._hcoeffs,
            optimize=True,
        )
        vvec = -(vals[:, 0] - vals[:, 1]) / 4
        return vvec

    @timeit
    def calc_v_pd(self):
        assert not self._model._localh
        np = len(self._params)
        vvec = numpy.zeros(np)
        if np == 0:
            return vvec

        vals = numpy.zeros((np, self._hsize))
        vals_width = numpy.zeros((np, self._hsize))
        vals_cost = numpy.zeros((np, self._hsize))

        if self._mrank == 0:
            print(f" calc_v_pd circs to run: {np * self._hsize}")

        rem = (np * self._hsize) % self._msize
        chunk_sizes = [
            (np * self._hsize) // self._msize + (1 if i < rem else 0)
            for i in range(self._msize)
        ]

        start = sum(chunk_sizes[: self._mrank])
        end = start + chunk_sizes[self._mrank]

        params_in_chunk = list(set([ind // self._hsize for ind in range(start, end)]))

        circs_in_chunk = dict()
        for param in params_in_chunk:
            circ = self._base_circs[param].copy()
            circ.apply_gates(
                self._poolpgates[self._ansatz_pids[param]],
                contract=False,
            )
            for k in self._ansatz_pids[param + 1 :]:
                circ.apply_gates(
                    self.get_poolrgate(k),
                    contract=False,
                )
            circs_in_chunk[param] = circ

        for _, ind in enumerate(range(start, end)):
            param_i = ind // self._hsize
            h_i = ind % self._hsize
            circ = circs_in_chunk[param_i].copy()

            op_gates = self._pstrings_h[h_i]

            circ.apply_gates(
                op_gates,
                contract=False,
            )
            for k in self._ansatz_pids[::-1]:
                circ.apply_gates(
                    self.get_poolrgate(k),
                    contract=False,
                )
            params_dict = circ.get_params()
            assert np * 2 == len(params_dict)
            for k, key in enumerate(params_dict):
                if k < np:
                    params_dict[key] = [self._params[k]]
                else:
                    params_dict[key] = [-self._params[2 * np - k - 1]]
            circ.set_params(params_dict)
            reh = circ.amplitude_rehearse(
                self._model._incar["ref_state"],
                optimize=self._config["optimize"],
                simplify_sequence=self._config["simplify_sequence"],
            )
            # if reh['W'] > 25:
            #     reh = circ.amplitude_rehearse(
            #             self._model._incar["ref_state"],
            #             optimize=opt_eco1,
            #             simplify_sequence=self._config["simplify_sequence"],
            #             )
            res = reh["tn"].contract(
                all,
                optimize=reh["tree"],
                output_inds=(),
                backend=self._config["backend"],
            )
            vals[param_i, h_i] = -res.imag / 2
            vals_width[param_i, h_i] = reh["W"]
            vals_cost[param_i, h_i] = reh["C"]

            if reh["W"] > self.max_tn_width:
                self.psave_circ(circ, "v_pd", reh["W"], reh["C"])
                self.max_tn_width = reh["W"]
                self.max_tn_cost = reh["C"]

        vals_allreduce = numpy.empty(numpy.shape(vals), dtype=float)
        self._comm.Allreduce([vals, MPI.DOUBLE], [vals_allreduce, MPI.DOUBLE])
        vals = vals_allreduce

        vals_width_reduce = numpy.empty(numpy.shape(vals_width), dtype=float)
        self._comm.Reduce(
            [vals_width, MPI.DOUBLE], [vals_width_reduce, MPI.DOUBLE], root=0
        )

        vals_cost_reduce = numpy.empty(numpy.shape(vals_cost), dtype=float)
        self._comm.Reduce(
            [vals_cost, MPI.DOUBLE], [vals_cost_reduce, MPI.DOUBLE], root=0
        )

        if self._mrank == 0:
            self.psave_tn_w_c(vals_width_reduce, vals_cost_reduce, "v_pd")

        vvec = vals.dot(self._hcoeffs)
        return vvec

    @timeit
    def calc_e(self):
        vals_h = numpy.zeros(self._hsize)

        if self._model._localh:
            if len(self._base_circs) > 0:
                circ = self._base_circs[-1].copy()
            else:
                circ = self._ref_circ.copy()

        if self._mrank == 0:
            print(f" calc_e circs to run: {self._hsize}")

        for i, op in enumerate(self._pstrings_h):
            if i % self._msize == self._mrank:
                if self._model._localh:
                    exp_val = circ.local_expectation(
                        op[0],
                        op[1],
                        optimize=self._config["optimize"],
                        simplify_sequence=self._config["simplify_sequence"],
                    )
                else:
                    exp_val = self.get_expectation_via_amplitude(
                        op, params=self._params
                    )[0]
                assert abs(exp_val.imag) < 1e-7
                vals_h[i] = exp_val.real

        vals_h_reduce = numpy.empty(numpy.shape(vals_h), dtype=float)
        self._comm.Reduce([vals_h, MPI.DOUBLE], [vals_h_reduce, MPI.DOUBLE], root=0)
        vals_h = vals_h_reduce

        if self._mrank == 0:
            self._vals_pstrings = vals_h
            res = self._hcoeffs.dot(vals_h).real
        else:
            res = None

        res = self._comm.bcast(res, root=0)
        self._comm.Barrier()

        return res

    def get_expectation_via_amplitude(self, op_gates, params=None):
        if len(self._base_circs) > 0:
            circ = self._base_circs[-1].copy()
        else:
            circ = self._ref_circ.copy()

        circ.apply_gates(
            op_gates,
            contract=False,
        )
        for i in self._ansatz_pids[::-1]:
            circ.apply_gates(
                self.get_poolrgate(i),
                contract=False,
            )
        params_dict = circ.get_params()
        if params is None:
            params = self._params
        np = len(params)
        assert np * 2 == len(params_dict)

        if np > 0:
            for i, key in enumerate(params_dict):
                if i < len(params):
                    params_dict[key] = [params[i]]
                else:
                    params_dict[key] = [-params[2 * np - i - 1]]
            circ.set_params(params_dict)
        reh = circ.amplitude_rehearse(
            self._model._incar["ref_state"],
            optimize=self._config["optimize"],
            simplify_sequence=self._config["simplify_sequence"],
        )
        # if reh['W'] > 25:
        #     reh = circ.amplitude_rehearse(
        #             self._model._incar["ref_state"],
        #             optimize=opt_eco1,
        #             simplify_sequence=self._config["simplify_sequence"],
        #             )
        res = reh["tn"].contract(
            all,
            optimize=reh["tree"],
            output_inds=(),
            backend=self._config["backend"],
        )

        if reh["W"] > self.max_tn_width:
            self.psave_circ(circ, "exp_val", reh["W"], reh["C"])
            self.max_tn_width = reh["W"]
            self.max_tn_cost = reh["C"]

        return res, reh["W"], reh["C"]

    @timeit
    def calc_e2(self):
        if self._mrank == 0:
            print(f" calc_e2 circs to run: {self._hhsize}")

        vals_hh = [
            self.get_expectation_via_amplitude(op)[0].real
            for op in self._pstrings_hhlocal
        ]

        vals_hh_final = numpy.zeros(self._hhsize)
        sendcountes = tuple(self.chunk_sizes_hh)
        displacements = tuple(self.shifts_hh)
        self._comm.Gatherv(
            [numpy.asarray(vals_hh), MPI.DOUBLE],
            [vals_hh_final, sendcountes, displacements, MPI.DOUBLE],
            root=0,
        )
        vals_hh = vals_hh_final

        if self._mrank == 0:
            vals_hh = numpy.hstack(vals_hh)
            self._vals_pstrings = numpy.append(self._vals_pstrings, vals_hh)
            res = self._hhcoeffs.dot(self._vals_pstrings[self._hhinds]).real
        else:
            res = None

        res = self._comm.bcast(res, root=0)
        self._comm.Barrier()

        return res

    @timeit
    def eval_vpool(self):
        vals_ph = self.calc_poolh()

        if self._mrank == 0:
            vals_pstrings = numpy.append(self._vals_pstrings, vals_ph)
            # get the complete new elements for vvec of dimension of the pool
            self._vvec_p = numpy.asarray(
                [
                    coeffs.dot(vals_pstrings[inds]).real
                    for coeffs, inds in zip(self._poolh[0], self._poolh[1])
                ]
            )
        else:
            self._vvec_p = None
        self._vvec_p = self._comm.bcast(self._vvec_p, root=0)
        self._comm.Barrier()

    @timeit
    def calc_poolh(self):
        if self._mrank == 0:
            print(f" calc_poolh circs to run: {self._phsize}")

        vals_ph = []
        width_ph = []
        cost_ph = []
        for op in self._pstrings_phlocal:
            vals, width, cost = self.get_expectation_via_amplitude(op)
            vals_ph.append(vals.real)
            width_ph.append(width)
            cost_ph.append(cost)

        vals_ph_final = numpy.zeros(self._phsize)
        width_ph_final = numpy.zeros(self._phsize)
        cost_ph_final = numpy.zeros(self._phsize)
        sendcountes = tuple(self.chunk_sizes_ph)
        displacements = tuple(self.shifts_ph)

        self._comm.Gatherv(
            [numpy.asarray(vals_ph), MPI.DOUBLE],
            [vals_ph_final, sendcountes, displacements, MPI.DOUBLE],
            root=0,
        )
        vals_ph = vals_ph_final
        self._comm.Gatherv(
            [numpy.asarray(width_ph), MPI.DOUBLE],
            [width_ph_final, sendcountes, displacements, MPI.DOUBLE],
            root=0,
        )
        self._comm.Gatherv(
            [numpy.asarray(cost_ph), MPI.DOUBLE],
            [cost_ph_final, sendcountes, displacements, MPI.DOUBLE],
            root=0,
        )
        if self._mrank == 0:
            vals_ph = numpy.hstack(vals_ph)
            self.psave_tn_w_c(width_ph_final, cost_ph_final, "pool_h", freq=1)

        return vals_ph

    @timeit
    def eval_mmat_pt(self):
        # off-diagonal block between pool and current ansatz.
        np = len(self._pool)
        nth = len(self._params)
        if self._mrank == 0:
            print(f" calc_mmat_pt circs to run: {np * nth}")

        mmat_pt = numpy.zeros((np, nth))
        mmat_width = numpy.zeros((np, nth))
        mmat_cost = numpy.zeros((np, nth))
        ij = 0
        for i in range(np):
            for j in range(nth):
                if ij % self._msize == self._mrank:
                    circ = self.get_mcirc(
                        nth,
                        j,
                        gate_i=self._poolpgates[i],
                    )
                    res, width, cost = self.calc_mmat_i(nth, circ)
                    mmat_pt[i, j] = res
                    mmat_width[i, j] = width
                    mmat_cost[i, j] = cost
                ij += 1

        self._mmat_pt = numpy.empty(numpy.shape(mmat_pt), dtype=float)
        self._comm.Allreduce([mmat_pt, MPI.DOUBLE], [self._mmat_pt, MPI.DOUBLE])
        mmat_width_reduce = numpy.empty(numpy.shape(mmat_width), dtype=float)
        self._comm.Reduce(
            [mmat_width, MPI.DOUBLE], [mmat_width_reduce, MPI.DOUBLE], root=0
        )
        mmat_cost_reduce = numpy.empty(numpy.shape(mmat_cost), dtype=float)
        self._comm.Reduce(
            [mmat_cost, MPI.DOUBLE], [mmat_cost_reduce, MPI.DOUBLE], root=0
        )

        if self._mrank == 0:
            self.psave_tn_w_c(mmat_width_reduce, mmat_cost_reduce, "mmat_pt", freq=1)

        if hasattr(self, "_mmat_pp"):
            # reset to ones
            self._mmat_pp[:] = 1
        else:
            self._mmat_pp = numpy.ones(np * (np - 1) // 2)  # to be evaluated adaptively
            self._mmat_pp_buf = self._mmat_pp.copy()

    def update_ansatz_circ_params(self):
        if len(self._base_circs) > 0:
            params_dict = self._base_circs[-1].get_params()
            assert len(params_dict) == len(self._params)
            for i, key in enumerate(params_dict):
                params_dict[key] = [self._params[i]]
            self._base_circs[-1].set_params(params_dict)

    def update_ngates(self, idx=-1):
        """
        update gate counts.
        """
        label = self._ansatz[idx]
        iorder = len(label) - label.count("I")
        self._ngates[iorder - 1] += 1

    @timeit
    def set_pstrings_hp(self):
        self.set_pstrings_h()
        self.set_pstrings_hh()
        self.set_poolh()
        self.quimbify_pstrings()

    def quimbify_pstrings(self):
        if self._model._localh:
            # to quimb pauli operators
            self._pstrings_h = [
                quimbify.pauli_string_to_quimb_op(p) for p in self._pstrings_h
            ]
        else:
            # to quimb gates
            self._pstrings_h = [
                quimbify.pauli_string_to_quimb_gates(p) for p in self._pstrings_h
            ]
        # generally nonlocal in qubits for the rest
        self._pstrings_hhlocal = [
            quimbify.pauli_string_to_quimb_gates(p) for p in self._pstrings_hhlocal
        ]
        self._pstrings_phlocal = [
            quimbify.pauli_string_to_quimb_gates(p) for p in self._pstrings_phlocal
        ]

    def set_pstrings_h(self):
        """
        Generates Pauli strings for h.
        """
        clabels = [clabel.split("*") for clabel in self._model._h]
        self._hcoeffs = numpy.asarray([float(clabel[0]) for clabel in clabels])
        hlabels = [clabel[1] for clabel in clabels]
        self._pstrings_h = hlabels
        self._hsize = len(hlabels)

    @timeit
    def set_pstrings_hh(self):
        """
        Generates Pauli strings for h^2 and redistributes them over MPI processes.
        """
        # employing Qiskit here
        self._hop = hop = SparsePauliOp(self._pstrings_h, self._hcoeffs)
        # simplify removes duplicates from the operator list
        # chop equates very small numbers to zero
        hhop = (hop @ hop).simplify().chop()
        self._hhcoeffs = hhop.coeffs.real
        hhlabels = [p.to_label() for p in hhop.paulis]

        pstrings = set(hhlabels)
        pstrings.difference_update(self._pstrings_h)
        self._pstrings_hh = list(pstrings)
        self._hhsize = len(self._pstrings_hh)

        # split and distribute pstrings
        rem = self._hhsize % self._msize
        self.chunk_sizes_hh = [
            self._hhsize // self._msize + (1 if i < rem else 0)
            for i in range(self._msize)
        ]
        self.shifts_hh = [sum(self.chunk_sizes_hh[:i]) for i in range(self._msize)]
        if self._mrank == 0:
            chunks = [
                self._pstrings_hh[
                    self.shifts_hh[i] : self.shifts_hh[i] + self.chunk_sizes_hh[i]
                ]
                for i in range(self._msize)
            ]
            print(f" pstrings-hh: {self._hhsize}")

            pstrings = numpy.concatenate((self._pstrings_h, self._pstrings_hh))
            sorted_inds = numpy.argsort(pstrings)
            pstrings = pstrings[sorted_inds]
            # numpy.searchsorted(pstrings, hhlabels) gives positions of hhlabels
            # within the pstrings
            # not sure why sorted_inds[] is here and what self._hhinds actually is
            self._hhinds = sorted_inds[numpy.searchsorted(pstrings, hhlabels)]
            assert self._hhinds.shape == self._hhcoeffs.shape
        else:
            chunks = None
        self._pstrings_hhlocal = self._comm.scatter(chunks, root=0)

    @timeit
    def set_poolh(self):
        """
        Generates Pauli strings for pool*h and redistributes them over MPI processes.
        """
        poolh = [[], []]
        pstrings = set()
        for label in self._pool:
            op = SparsePauliOp(label)
            oph = (op @ self._hop).simplify().chop()
            oph *= -0.5j
            poolh[0].append(oph.coeffs.real)
            poolh[1].append([p.to_label() for p in oph.paulis])
            pstrings.update(poolh[1][-1])
        pstrings.difference_update(self._pstrings_h)
        pstrings.difference_update(self._pstrings_hh)
        self._pstrings_ph = list(pstrings)
        self._phsize = len(self._pstrings_ph)
        # print(f" pstrings-ph: {self._phsize}")

        pstrings = numpy.concatenate(
            (
                self._pstrings_h,
                self._pstrings_hh,
                self._pstrings_ph,
            )
        )
        sorted_inds = numpy.argsort(pstrings)
        pstrings = pstrings[sorted_inds]
        poolh[1] = [
            sorted_inds[numpy.searchsorted(pstrings, oplabels)] for oplabels in poolh[1]
        ]
        self._poolh = poolh

        # split and distribute pstrings
        rem = self._phsize % self._msize
        self.chunk_sizes_ph = [
            self._phsize // self._msize + (1 if i < rem else 0)
            for i in range(self._msize)
        ]
        self.shifts_ph = [sum(self.chunk_sizes_ph[:i]) for i in range(self._msize)]

        if self._mrank == 0:
            chunks = [
                self._pstrings_ph[
                    self.shifts_ph[i] : self.shifts_ph[i] + self.chunk_sizes_ph[i]
                ]
                for i in range(self._msize)
            ]
        else:
            self._pstrings_ph = chunks = None

        self._pstrings_phlocal = self._comm.scatter(chunks, root=0)

        if self._mrank == 0:
            del self._pstrings_ph, self._pstrings_hh

    def get_mv(
        self,
        inds,  # indices of additonal unitaries to be appended
        parallel=False,
    ):
        ntheta = self._vvec.shape[0]
        ninds = len(inds)
        vvec = numpy.zeros(ntheta + ninds)
        if ntheta > 0:
            vvec[:ntheta] = self._vvec

        # add vvec
        vvec[ntheta:] = self._vvec_p[inds]

        if parallel:
            self._mmat_pp_buf *= 0

        mmat = numpy.zeros((ntheta + ninds, ntheta + ninds))
        icount = 0
        for i, ind_i in enumerate(inds):
            for shift, inds_cur in zip(
                [self._ntheta_old, ntheta], [self._inds_add, inds]
            ):
                for j, ind_j in enumerate(inds_cur):
                    if shift == ntheta and i == j:
                        break
                    iu, jl = max(ind_i, ind_j), min(ind_i, ind_j)
                    ij = iu * (iu - 1) // 2 + jl
                    if self._mmat_pp[ij] > 0.5:  # havenot been evaluated
                        if (
                            parallel and icount % self._msize == self._mrank
                        ) or not parallel:
                            circ = self.get_mcirc(
                                self._ntheta_old,
                                self._ntheta_old - 1,
                                gate_i=self._poolpgates[iu],
                                gate_j=self._poolpgates[jl],
                            )
                            res, width, cost = self.calc_mmat_i(self._ntheta_old, circ)
                            if abs(res) < 1e-16:  # for marking purpose
                                res = 1e-16
                            self._mmat_pp_buf[ij] = mmat[ntheta + i, shift + j] = mmat[
                                shift + j, ntheta + i
                            ] = res
                        icount += 1
                    else:
                        if (
                            parallel and icount % self._msize == self._mrank
                        ) or not parallel:
                            mmat[ntheta + i, shift + j] = mmat[
                                shift + j, ntheta + i
                            ] = self._mmat_pp[ij]

        if parallel:
            mmat_pp_buf_allreduce = numpy.empty(
                numpy.shape(self._mmat_pp_buf), dtype=float
            )
            self._comm.Allreduce(
                [self._mmat_pp_buf, MPI.DOUBLE], [mmat_pp_buf_allreduce, MPI.DOUBLE]
            )
            self._mmat_pp_buf = mmat_pp_buf_allreduce

            mmat_allreduce = numpy.empty(numpy.shape(mmat), dtype=float)
            self._comm.Allreduce([mmat, MPI.DOUBLE], [mmat_allreduce, MPI.DOUBLE])
            mmat = mmat_allreduce

            where = numpy.where(abs(self._mmat_pp_buf) > 1e-17)[0]
            self._mmat_pp[where] = self._mmat_pp_buf[where]

        numpy.fill_diagonal(mmat, 0.25)
        mmat[:ntheta, :ntheta] = self._mmat

        # add off-diagonal columns for mmat
        for i, ind_i in enumerate(inds):
            mmat[ntheta + i, : self._ntheta_old] = mmat[
                : self._ntheta_old, ntheta + i
            ] = self._mmat_pt[ind_i, :]
        assert ntheta - self._ntheta_old == len(self._inds_add)
        return mmat, vvec
