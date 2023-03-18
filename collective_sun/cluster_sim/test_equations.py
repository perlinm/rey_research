import itertools
import pytest

import operators as ops
import equations as eqs

import numpy as np
import sparse
import scipy


np.random.seed(0)
np.set_printoptions(linewidth=200, precision=3)

QUBIT_OP_MATS = ops.get_qubit_op_mats()


@pytest.mark.parametrize("num_sites", [3])
def test_op_poly(num_sites: int) -> None:
    op_mat = ops.get_random_op(2**num_sites)
    op_poly = eqs.OperatorPolynomial.from_matrix(op_mat, QUBIT_OP_MATS)

    for term, _ in op_poly.factorize(eqs.trivial_factorizer):
        assert term.num_factors() == 1 or term.is_empty()

    for term, _ in op_poly.factorize(eqs.mean_field_factorizer):
        for op, exp in term:
            assert op.locality == exp == 1


@pytest.mark.parametrize("num_sites", [3])
def test_exact_spin_model(num_sites: int) -> None:
    dim = 2**num_sites
    structure_factors = ops.get_structure_factors(*QUBIT_OP_MATS)

    # build a random Hamiltonian
    ham_mat = ops.get_random_op(dim, hermitian=True)
    hamiltonian = ops.DenseMultiBodyOperators.from_matrix(ham_mat, QUBIT_OP_MATS)

    # collect a sequence of single-body operators (by index) into a single multi-body operator
    def multi_body_op(*local_ops: int):
        op_list = [
            ops.SingleBodyOperator(local_op, site)
            for local_op, site in zip(local_ops, range(num_sites))
        ]
        return ops.MultiBodyOperator(*op_list)

    # build time derivative tensors
    all_ops = [
        multi_body_op(*local_ops) for local_ops in itertools.product(range(4), repeat=num_sites)
    ]
    op_to_index, time_deriv_tensors = eqs.build_equations_of_motion(
        *all_ops,
        hamiltonian=hamiltonian,
        structure_factors=structure_factors,
    )
    op_product_to_index = {
        eqs.ExpectationValueProduct(op): index for op, index in op_to_index.items()
    }

    if not hasattr(sparse, "einsum"):
        # this sparse tensor library version does not support einsum, so convert to numpy arrays
        time_deriv_tensors = tuple(tensor.todense() for tensor in time_deriv_tensors)

    # construct a random operator
    op_mat = ops.get_random_op(dim)
    op_poly = eqs.OperatorPolynomial.from_matrix(op_mat, QUBIT_OP_MATS)
    op_vec = op_poly.to_array(op_product_to_index)

    # time-evolve the random operator
    solution = scipy.integrate.solve_ivp(
        eqs.time_deriv,
        [0, 1],
        op_vec,
        t_eval=[1],
        args=(time_deriv_tensors,),
    )
    final_vec = solution.y[:, -1]

    # time-evolve the random operator by brute-force Heisenberg evolution
    iden = np.eye(dim)
    ham_gen: np.ndarray = np.kron(ham_mat, iden) - np.kron(iden, ham_mat.T)
    expected_solution = scipy.integrate.solve_ivp(
        lambda _, vec, generator: 1j * (generator @ vec),
        [0, 1],
        op_mat.ravel(),
        t_eval=[1],
        args=(ham_gen,),
    )
    expected_final_mat = expected_solution.y[:, -1].reshape((dim, dim))
    expected_final_poly = eqs.OperatorPolynomial.from_matrix(expected_final_mat, QUBIT_OP_MATS)
    expected_final_vec = expected_final_poly.to_array(op_product_to_index)

    assert np.allclose(final_vec, expected_final_vec, atol=1e-3)
