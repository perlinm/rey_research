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
INTEGRATION_OPTIONS = dict(method="DOP853", rtol=1e-8)


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


@pytest.mark.parametrize("num_sites", [10])
def test_mean_field(num_sites: int) -> None:
    local_dim = 2
    local_op_mats = QUBIT_OP_MATS
    structure_factors = ops.get_structure_factors(*local_op_mats)

    def get_random_coupling_matrix() -> np.ndarray:
        return ops.get_random_matrix(num_sites, hermitian=True, diagonal=True, real=True)

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
        factorization_rule=eqs.mean_field_factorizer,
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
    expected_solution = scipy.integrate.solve_ivp(
        time_deriv_MF,
        [0, 1],
        init_state.astype(complex).ravel(),
        t_eval=[1],
        args=(hamiltonian, np.array(local_op_mats)),
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
    hamiltonian: ops.DenseMultiBodyOperators,
    op_mats: np.ndarray,
) -> np.ndarray:
    num_sites = hamiltonian.terms[0].num_sites
    state.shape = (num_sites, -1)
    op_state_vecs = np.einsum("oij,sj->soi", op_mats, state)
    op_exp_vals = np.einsum("si,soi->so", state.conj(), op_state_vecs)
    state.shape = (-1,)
    time_deriv = sum(
        (_time_deriv_MF(op_state_vecs, op_exp_vals, term) for term in hamiltonian.terms),
        start=np.array(0),
    )
    return time_deriv.ravel()


def _time_deriv_MF(
    op_state_vecs: np.ndarray,
    op_exp_vals: np.ndarray,
    hamiltonian: ops.DenseMultiBodyOperator,
) -> np.ndarray:
    num_sites, _, local_dim = op_state_vecs.shape

    fixed_ops = hamiltonian.fixed_ops
    dist_ops = hamiltonian.dist_ops
    fixed_sites = hamiltonian.fixed_sites
    dist_sites = [
        site for site in ops.LatticeSite.range(num_sites) if site not in hamiltonian.fixed_sites
    ]
    tensor = hamiltonian.scalar * hamiltonian.tensor

    def term_to_time_deriv_and_scalar(
        op_sites: tuple[ops.LatticeSite, ...],
        local_ops: tuple[ops.AbstractSingleBodyOperator, ...],
    ) -> tuple[np.ndarray, complex]:
        site_indices = np.array([site.index for site in op_sites], dtype=int)
        local_op_indices = np.array([local_op.index for local_op in local_ops], dtype=int)
        scalars = op_exp_vals[site_indices, local_op_indices]
        time_deriv = op_state_vecs[site_indices, local_op_indices, :]
        for idx, scalar in enumerate(scalars):
            other_indices = np.array([ii for ii in range(len(op_sites)) if ii != idx], dtype=int)
            time_deriv[other_indices, :] *= scalar
        return time_deriv, np.prod(scalars, dtype=complex)

    # identify contributions from fixed ops
    fixed_time_deriv, fixed_op_scalar = term_to_time_deriv_and_scalar(fixed_sites, fixed_ops)

    # identify contributions from distributed ops
    dist_time_deriv = np.zeros((num_sites, local_dim), dtype=complex)
    dist_op_scalar = complex(1.0)
    for addressed_sites in itertools.combinations(dist_sites, len(dist_ops)):
        for sites in itertools.permutations(addressed_sites):
            term_time_deriv, term_scalar = term_to_time_deriv_and_scalar(sites, dist_ops)
            dist_op_scalar += tensor[sites] * term_scalar
            dist_time_deriv[np.array(sites, dtype=int), :] += tensor[sites] * term_time_deriv

    time_deriv = fixed_op_scalar * dist_time_deriv
    time_deriv[np.array(fixed_sites, dtype=int), :] += dist_op_scalar * fixed_time_deriv
    return -1j * time_deriv
