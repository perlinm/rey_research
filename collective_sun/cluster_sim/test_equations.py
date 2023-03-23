import numpy as np
import pytest
import scipy
import sparse

import equations as eqs
import operators as ops

np.random.seed(0)
np.set_printoptions(linewidth=200, precision=3)

INTEGRATION_OPTIONS = dict(method="DOP853", rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("num_sites", [3])
def test_op_poly(num_sites: int) -> None:
    op_mat = ops.get_random_matrix(2**num_sites)
    op_poly = eqs.OperatorPolynomial.from_matrix(op_mat, ops.get_qubit_op_mats())

    for term, _ in op_poly.factorize(eqs.trivial_factorizer):
        assert term.num_factors() == 1 or term.is_empty()

    for term, _ in op_poly.factorize(eqs.mean_field_factorizer):
        for op, exp in term:
            assert op.locality == exp == 1


@pytest.mark.parametrize("num_sites, local_dim", [(3, 2), (3, 3)])
def test_spin_model(num_sites: int, local_dim: int) -> None:
    local_op_mats = ops.get_spin_qudit_op_mats(local_dim)
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
        factorization_rule=eqs.trivial_factorizer,
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
        **INTEGRATION_OPTIONS,
    )
    final_vec = solution.y[:, -1]

    # time-evolve the random operator by "brute force"
    iden = np.eye(dim)
    ham_gen: np.ndarray = np.kron(ham_mat, iden) - np.kron(iden, ham_mat.T)
    expected_solution = scipy.integrate.solve_ivp(
        lambda _, vec, generator: -1j * (generator @ vec),
        [0, 1],
        init_op_mat.ravel(),
        t_eval=[1],
        args=(ham_gen,),
        **INTEGRATION_OPTIONS,
    )
    expected_final_mat = expected_solution.y[:, -1].reshape((dim, dim))
    expected_final_poly = eqs.OperatorPolynomial.from_matrix(expected_final_mat, local_op_mats)
    expected_final_vec = expected_final_poly.to_array(op_to_index)

    assert np.allclose(final_vec, expected_final_vec, atol=1e-4)


def cumulant_mean_field_factorizer(op: ops.MultiBodyOperator) -> eqs.OperatorPolynomial:
    return eqs.cumulant_factorizer(op, lambda _: False)


@pytest.mark.parametrize(
    "num_sites, local_dim, factorization_rule",
    [
        (10, dim, factorizer)
        for dim in [2, 3]
        for factorizer in [eqs.mean_field_factorizer, cumulant_mean_field_factorizer]
    ],
)
def test_mean_field(
    num_sites: int, local_dim: int, factorization_rule: eqs.FactorizationRule
) -> None:
    local_op_mats = ops.get_spin_qudit_op_mats(local_dim)
    structure_factors = ops.get_structure_factors(*local_op_mats)

    def get_random_coupling_matrix() -> np.ndarray:
        return ops.get_random_matrix(num_sites, real=True, off_diagonal=True)

    # build a random 2-body Hamiltonian
    ham_terms = [
        ops.DenseMultiBodyOperator(op_a, op_b, tensor=get_random_coupling_matrix())
        for op_a in ops.AbstractSingleBodyOperator.range(local_dim**2)
        for op_b in ops.AbstractSingleBodyOperator.range(local_dim**2)
    ]
    hamiltonian = ops.DenseMultiBodyOperators(*ham_terms)

    # construct a Haar-random initial product state, indexed by (site, local_state)
    init_state = scipy.stats.unitary_group.rvs(local_dim, size=num_sites)[:, :, 0]
    init_op_poly = eqs.OperatorPolynomial.from_product_state(init_state, local_op_mats)

    # build time derivative tensors
    op_to_index, time_deriv_tensors = eqs.build_equations_of_motion(
        *init_op_poly.vec.keys(),
        hamiltonian=hamiltonian,
        structure_factors=structure_factors,
        factorization_rule=factorization_rule,
    )

    if not hasattr(sparse, "einsum"):
        # this sparse tensor library version does not support einsum, so convert to numpy arrays
        time_deriv_tensors = tuple(tensor.todense() for tensor in time_deriv_tensors)

    # time-evolve the random initial state
    init_op_vec = init_op_poly.to_array(op_to_index)
    solution = scipy.integrate.solve_ivp(
        eqs.time_deriv,
        [0, 1],
        init_op_vec,
        t_eval=[1],
        args=(time_deriv_tensors,),
        **INTEGRATION_OPTIONS,
    )
    final_vec = solution.y[:, -1]

    # time-evolve the random initial state by "brute force"
    ham_tensors = tuple(term.to_coefficient_tensor(local_op_mats) for term in hamiltonian.terms)
    localities = tuple(term.locality for term in hamiltonian.terms)
    expected_solution = scipy.integrate.solve_ivp(
        time_deriv_MF,
        [0, 1],
        init_state.astype(complex).ravel(),
        t_eval=[1],
        args=(ham_tensors, localities, num_sites),
        **INTEGRATION_OPTIONS,
    )
    final_state_MF = expected_solution.y[:, -1].reshape((num_sites, local_dim))
    expected_final_vec = np.zeros_like(final_vec)
    for multi_body_op, idx in op_to_index.items():
        if multi_body_op.is_identity_op():
            expected_final_vec[idx] = 1
            continue
        local_op = next(iter(multi_body_op.ops))
        op_mat = local_op_mats[local_op.op]
        local_state = final_state_MF[local_op.site, :]
        expected_final_vec[idx] = local_state.conj() @ op_mat @ local_state

    assert np.allclose(final_vec, expected_final_vec, atol=1e-4)


def time_deriv_MF(
    _: float,
    state: np.ndarray,
    ham_tensors: tuple[np.ndarray, ...],
    localities: tuple[int, ...],
    num_sites: int,
) -> np.ndarray:
    state.shape = (num_sites, -1)
    time_deriv = sum(
        (
            _time_deriv_MF(state, tensor, locality)
            for tensor, locality in zip(ham_tensors, localities)
        ),
        start=np.array(0),
    )
    state.shape = (-1,)
    return time_deriv.ravel()


def _time_deriv_MF(
    state: np.ndarray,
    ham_tensor: np.ndarray,
    locality: int,
) -> np.ndarray:
    if locality == 0:
        return -1j * ham_tensor * state
    state_pairs = [state.conj(), state] * (locality - 1)
    sites_indices = "abcdefghijklmnopqrstuvwxyz"[:locality]
    op_indices = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: 2 * locality]
    tensor_indices = sites_indices + op_indices
    state_indices = [sites_indices[ii // 2] + op_indices[ii : ii + 1] for ii in range(2 * locality)]
    state_index_expr = ",".join(state_indices[:-2] + [state_indices[-1]])
    index_expr = f"{tensor_indices},{state_index_expr}->{state_indices[-2]}"
    return -1j * locality * np.einsum(index_expr, ham_tensor, *state_pairs, state)
