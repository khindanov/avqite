[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15531600.svg)](https://doi.org/10.5281/zenodo.15531600)

# avqite
Tensor-network implementation of the Adaptive Variational Quantum Imaginary-Time Evolution (AVQITE) algorithm using Quimb. The code is parallelized using MPI.

## Authors

- **Yongxin Yao** ([@yaoyongxin](https://github.com/yaoyongxin))
- **Aleksei Khindanov** ([@khindanov](https://github.com/khindanov))

## Prerequisites

Before installing Python packages, ensure you have the following prerequisites:

It is recommended to use a **virtual environment**:

1. Using **venv** (built-in):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  
   
   # On Windows: .venv\Scripts\activate
   ```

2. Using **conda**

   - Install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html)

   ```bash
   conda create -n myenv python=3.10
   conda activate myenv
   ```

### MPI (mpi4py) Installation

This package uses **MPI (mpi4py)** for parallel execution. See official **mpi4py** [installation page](https://mpi4py.readthedocs.io/en/4.0.3/install.html).


- **Linux/macOS**: It is recommended to use **conda-forge** with the desired MPI implementation, for example MPICH:
  ```bash
  conda install -c conda-forge mpi4py mpich
  ```
  Alternatively, one can try installing via **pip**, but installation problems can be encountered in this case:
  
  ```bash
  python -m pip install mpi4py
  ```

- **Windows**: Install Microsoft MPI (MS-MPI)
  1. Download and install both the MS-MPI SDK and runtime from the [Microsoft website](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi)
  2. Add the MPI installation directory to your system PATH

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/khindanov/avqite.git
   ```

2. Create and activate a new environment (optional)

3. Install **project dependencies** (required):
   ```bash
   pip install -e .
   ```

4. Install **optional dependecies**:
   ```bash
   # For Windows users 
   pip install .[cotengra_advanced,mpi]

   # For Linux/macOS users
   pip install .[cotengra_advanced,kahypar]

   # For Zsh shell users
   pip install ".[cotengra_advanced,kahypar]"
   ```

### Important Notes

- Currently `mpi4py` is required to run the code. However, installing `mpi4py` via `pip install mpi4py` may fail on **macOS**. For this reason `mpi4py` is listed in optional dependencies, and **macOS** are recommended to install `mpi4py` using `conda-forge` (see above)
- **Windows Users**: The `kahypar` package (which is an optional dependency for `cotengra` used to perform optimized tensor network contractions) is not supported on Windows

## Running the Code

1. Activate environment:
   ```bash
   conda activate myenv
   ```

2. Execution:
   - 1D TFIM example
      ```bash
      mpiexec -n <number_of_processes> python examples/1dtfim/run.py --filename N10g0.1
      ```
   - 1D $Z_2$ LGT example
      ```bash
      mpiexec -n <number_of_processes> python examples/1dlgt/run.py --filename L9Zbasis0hz0seed --notetras -m 100 --tf 1.0
      ```
