import numpy as np
import json
from more_itertools import distinct_permutations


def build_hamiltonian(L, mu, bz):
    """
    Write system Hamiltonian as a list of Pauli strings
    """

    # Set string of identities
    id_str = L * 'I'

    # Add X term
    h_str = [f"0.25*{id_str[:i]+'X'+id_str[i+1:]}" for i in range(1, L-1)]

    # Add ZXZ term
    h_str.extend([f"-0.25*"+p*'I'+"ZXZ"+(L-3-p)*'I' for p in range(L - 2)])
        
    # Add chemical potential
    h_str.extend([f"{0.5 * mu}*{id_str[:i]+'ZZ'+id_str[i+2:]}" for i in range(L - 1)])

    # Add Z term (if non-zero)
    if bz:
        h_str.extend([f"{-bz}*{id_str[:i]+'Z'+id_str[i+1:]}" for i in range(L)])

    return h_str


def generate_pool(L, pool_set):
    '''
    Generate all possible distinct combinations of the strings in pool_set,
    filling up to system size with identity matrices
    '''

    comb = []
    for elem in pool_set:
        neye = L - len(elem)
        series = elem + neye * 'I'
        comb += [''.join(p) for p in distinct_permutations(series)]

    return comb


if __name__ == '__main__':
    for L in [9,17,25,33,41,49,57]:
        for seed in range(30,60):
            # Set system size

            # Define Hamiltonian parameters
            # bz = 0.1
            # mu = -0.53
            bz = 0
            mu = -0.7

            # Get Hamiltonian
            h = build_hamiltonian(L, mu, bz)

            # Set basis
            basis = 'Z'

            # Define set of pool operators
            if basis == 'Z':
                pool_set = ['Y', "YZ"]
            elif basis == 'Y':
                pool_set = ['X', 'Z', "ZX", "ZZZ"]

            # Get operator pool
            pool = generate_pool(L, pool_set)

            # Generate random product state
            ref_state = f"{''.join('0' if np.random.rand() > 0.5 else '1' for _ in range(L))}" 

            # Get JSON object
            dct = json.dumps(
                {
                    'h': h,
                    "pool": pool,
                    "ref_state": ref_state,
                },
                    indent=4)

            # Write incar file
            with open(f"incarL{L}{basis}basis{bz}hz{seed}seed", 'w') as f:
                f.write(dct)
