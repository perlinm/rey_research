import pytest

import operators as ops
import equations as eqs

import numpy as np


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


def test_qubit_EOM() -> None:
    op_Z = ops.MultiBodyOperator(ops.SingleBodyOperator(1, 0))
    op_X = ops.MultiBodyOperator(ops.SingleBodyOperator(2, 0))
    op_Y = ops.MultiBodyOperator(ops.SingleBodyOperator(3, 0))
    structure_factors = ops.get_structure_factors(*QUBIT_OP_MATS)

    op_seed = op_X
    ham_op = ops.DenseMultiBodyOperator(scalar=0.5, fixed_op=op_Z, num_sites=1)
    op_to_index, time_deriv_tensors = eqs.build_equations_of_motion(
        op_seed, ham_op, structure_factors
    )
    assert len(time_deriv_tensors) == 1

    order, tensor = next(iter(time_deriv_tensors.items()))
    assert order == 1

    expected_tensor = np.zeros((2, 2))
    expected_tensor[op_to_index[op_Y], op_to_index[op_X]] = 1
    expected_tensor[op_to_index[op_X], op_to_index[op_Y]] = -1
    assert np.allclose(expected_tensor, tensor.todense())
