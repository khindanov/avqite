import quimb
import quimb.tensor as qtn
import warnings



def add_pauli_rotation_gate(
    qc: "quimb.tensor.circuit.Circuit",
    pauli_string: str,
    theta: float = 0,
    decompose_rzz: bool = True
):
    """
    Appends a Pauli rotation gate to a Quimb Circuit.
    Convention for Pauli string ordering is opposite to the Qiskit convention.
    For example, in string "XYZ" Pauli "X" acts on the first qubit.

    Parameters
    ----------
    qc : "quimb.tensor.circuit.Circuit"
        Quimb Circuit to which the Pauli rotation gate is appended.
    pauli_string : str
        Pauli string defining the rotation.
    theta : float
        Rotation angle.
    decompose_rzz : bool
        If decompose_rzz==True, all rzz gates are decompsed into cx-rz-cx.
        Otherwise, the final circuit contains rzz gates.

    Returns
    -------
    qc: Parameterized "quimb.tensor.circuit.Circuit"
    """

    if qc.N != len(pauli_string):
        raise ValueError("Circuit and Pauli string are of different size")
    if all([pauli=='I' or pauli=='X' or pauli=='Y' or pauli=='Z'
            for pauli in pauli_string])==False:
        raise ValueError("Pauli string does not have a correct format")

    nontriv_pauli_list = [(i,pauli)
                        for i,pauli in enumerate(pauli_string) if pauli!='I']
    if len(nontriv_pauli_list) == 1:
        # RX, RY, or RZ
        assert(nontriv_pauli_list[0][1] in ['X', 'Y', 'Z'])
        gate = 'R' + nontriv_pauli_list[0][1]
        qc.apply_gate(
            gate,
            theta,
            nontriv_pauli_list[0][0],
            parametrize=True,
            gate_opts={'contract': False}
        )
    elif (len(nontriv_pauli_list) == 2 and
            nontriv_pauli_list[0][1] + nontriv_pauli_list[1][1] in ['XX', 'YY']):
        # RXX or RYY
        gate = 'R' + nontriv_pauli_list[0][1] + nontriv_pauli_list[1][1]
        qc.apply_gate(
            gate,
            theta,
            nontriv_pauli_list[0][0],
            nontriv_pauli_list[1][0],
            parametrize=True,
            gate_opts={'contract': False}
        )
    else:
        for (i,pauli) in nontriv_pauli_list:
            if pauli=='X':
                qc.apply_gate('H',i)
            if pauli=='Y':
                qc.apply_gate('SDG',i)
                qc.apply_gate('H',i)
        for list_ind in range(len(nontriv_pauli_list)-2):
            qc.apply_gate(
                'CX',
                nontriv_pauli_list[list_ind][0],
                nontriv_pauli_list[list_ind+1][0]
            )
        if decompose_rzz:
            qc.apply_gate(
                'CX',
                nontriv_pauli_list[len(nontriv_pauli_list)-2][0],
                nontriv_pauli_list[len(nontriv_pauli_list)-1][0]
            )
            qc.apply_gate(
                'RZ',
                theta,
                nontriv_pauli_list[len(nontriv_pauli_list)-1][0],
                parametrize=True,
                gate_opts={'contract': False}
            )
            qc.apply_gate(
                'CX',
                nontriv_pauli_list[len(nontriv_pauli_list)-2][0],
                nontriv_pauli_list[len(nontriv_pauli_list)-1][0]
            )
        else:
            qc.apply_gate(
                'RZZ',
                theta,
                nontriv_pauli_list[len(nontriv_pauli_list)-2][0],
                nontriv_pauli_list[len(nontriv_pauli_list)-1][0],
                parametrize=True,
                gate_opts={'contract': False}
            )
        for list_ind in reversed(range(len(nontriv_pauli_list)-2)):
            qc.apply_gate(
                'CX',
                nontriv_pauli_list[list_ind][0],
                nontriv_pauli_list[list_ind+1][0]
            )
        for (i,pauli) in nontriv_pauli_list:
            if pauli=='X':
                qc.apply_gate('H',i)
            if pauli=='Y':
                qc.apply_gate('H',i)
                qc.apply_gate('S',i)
    return qc


def pauli_string_to_quimb_gates(
        pauli_string: str
):
    """
    Converts a Pauli string into a Quimb gate.
    """
    gates = ()
    pauli_gates = ['X', 'Y', 'Z']
    for i, el in enumerate(pauli_string):
        if el in pauli_gates:
            gates += (qtn.circuit.Gate(
                label=el,
                params=[],
                qubits=(i,)
            ),)
        else:
            assert(el == "I")
    return gates


def pauli_string_to_quimb_op(
        pauli_string: str,
        ):
    weight = len(pauli_string) - pauli_string.count("I")
    if weight > 10:
        warnings.warn("qarray large for weight: {weight}")
    elif weight == 0:
        return 1, None
    op = None
    where = []
    for i, p in enumerate(pauli_string):
        if p == "I":
            continue
        where.append(i)
        if op is None:
            op = quimb.pauli(p)
        else:
            op = op & quimb.pauli(p)
    return op, where


if __name__ == "__main__":
    nq = 60
    params = [1, 2, 1]
    pstrings = ["XXYYII"+"I"*(nq-6), "XXYYII"+"I"*(nq-6), "IIYYII"+"I"*(nq-6)]
    circs = [add_pauli_rotation_gate(
        qc=qtn.Circuit(N=nq),
        pauli_string=p,
        theta=params[i],
        decompose_rzz=False,
        ) for i, p in enumerate(pstrings)
        ]
    print(circs)
    gates = [pauli_string_to_quimb_gates(p) for p in pstrings]
    print(gates)
