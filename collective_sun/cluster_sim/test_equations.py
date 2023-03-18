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


def get_multi_body_op(*local_ops: int) -> ops.MultiBodyOperator:
    """Place the given local operators on lattice sites 0, 1, 2, etc."""
    op_list = [
        ops.SingleBodyOperator(local_op, site)
        for local_op, site in zip(local_ops, range(len(local_ops)))
    ]
    return ops.MultiBodyOperator(*op_list)


def get_all_ops(num_sites: int, dim: int) -> tuple[ops.MultiBodyOperator, ...]:
    """Get all operations on a given number lattice sites with a given local dimension."""
    return tuple(
        get_multi_body_op(*local_ops)
        for local_ops in itertools.product(range(dim**2), repeat=num_sites)
    )


@pytest.mark.parametrize("num_sites", [3])
def test_op_poly(num_sites: int) -> None:
    op_mat = ops.get_random_matrix(2**num_sites)
    op_poly = eqs.OperatorPolynomial.from_matrix(op_mat, QUBIT_OP_MATS)

    for term, _ in op_poly.factorize(eqs.trivial_factorizer):
        assert term.num_factors() == 1 or term.is_empty()

    for term, _ in op_poly.factorize(eqs.mean_field_factorizer):
        for op, exp in term:
            assert op.locality == exp == 1


@pytest.mark.parametrize("num_sites", [3])
def test_spin_model(num_sites: int) -> None:
    local_dim = 2
    local_op_mats = QUBIT_OP_MATS
    dim = local_dim**num_sites
    structure_factors = ops.get_structure_factors(*local_op_mats)

    # build a random Hamiltonian
    ham_mat = ops.get_random_matrix(dim, hermitian=True)
    hamiltonian = ops.DenseMultiBodyOperators.from_matrix(ham_mat, local_op_mats)

    # construct a random operator
    init_op_mat = ops.get_random_matrix(dim)
    init_op_poly = eqs.OperatorPolynomial.from_matrix(init_op_mat, local_op_mats)

    # build time derivative tensors
    op_to_index, time_deriv_tensors = eqs.build_equations_of_motion(
        *init_op_poly.vec.keys(),
        hamiltonian=hamiltonian,
        structure_factors=structure_factors,
    )

    if not hasattr(sparse, "einsum"):
        # this sparse tensor library version does not support einsum, so convert to numpy arrays
        time_deriv_tensors = tuple(tensor.todense() for tensor in time_deriv_tensors)

    # time-evolve the random operator
    init_op_vec = init_op_poly.to_array(op_to_index)
    solution = scipy.integrate.solve_ivp(
        eqs.time_deriv,
        [0, 1],
        init_op_vec,
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
        init_op_mat.ravel(),
        t_eval=[1],
        args=(ham_gen,),
    )
    expected_final_mat = expected_solution.y[:, -1].reshape((dim, dim))
    expected_final_poly = eqs.OperatorPolynomial.from_matrix(expected_final_mat, local_op_mats)
    expected_final_vec = expected_final_poly.to_array(op_to_index)

    assert np.allclose(final_vec, expected_final_vec, atol=1e-3)


@pytest.mark.parametrize("num_sites", [10])
def test_mean_field(num_sites: int) -> None:
    local_dim = 2
    local_op_mats = QUBIT_OP_MATS
    dim = local_dim**num_sites
    structure_factors = ops.get_structure_factors(*local_op_mats)

    def get_random_coupling_matrix() -> np.ndarray:
        return ops.get_random_matrix(dim, hermitian=True, diagonal=True, real=True)

    # build a random 2-body Hamiltonian
    ham_terms = [
        ops.DenseMultiBodyOperator(op_a, op_b, tensor=get_random_coupling_matrix())
        for op_a in ops.AbstractSingleBodyOperator.range(local_dim**2)
        for op_b in ops.AbstractSingleBodyOperator.range(local_dim**2)
    ]
    ham_op = ops.DenseMultiBodyOperators(*ham_terms)

    # construct a Haar-random initial product state, indexed by (site, local_state)
    init_state = scipy.stats.unitary_group.rvs(local_dim, size=num_sites)[:, :, 0]
    init_op_poly = eqs.OperatorPolynomial.from_product_state(init_state, local_op_mats)

    structure_factors
    ham_op
    init_op_poly
