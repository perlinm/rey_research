import pytest

import operators as ops
import equations as eqs


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
def test_EOM(num_sites: int) -> None:
    ham_mat = ops.get_random_op(2**num_sites)
    ham_op = ops.DenseMultiBodyOperators.from_matrix(ham_mat, QUBIT_OP_MATS)
    op_seed = ops.MultiBodyOperator(ops.SingleBodyOperator(op=1, site=0))
    structure_factors = ops.get_structure_factors(*QUBIT_OP_MATS)

    eqs.build_equations_of_motion(op_seed, ham_op, structure_factors)
