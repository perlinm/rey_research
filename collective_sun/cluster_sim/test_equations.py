import pytest

import operators as ops
import equations as eqs


qubit_op_mats = ops.get_qubit_op_mats()


@pytest.mark.parametrize("num_sites", [3])
def test_op_poly(num_sites: int) -> None:
    op_mat = ops.get_random_op(2**num_sites)
    op_sum = ops.DenseMultiBodyOperators.from_matrix(op_mat, qubit_op_mats)
    op_poly = eqs.OperatorPolynomial.from_dense_ops(op_sum)

    # op_poly = op_poly.factorize(eqs.trivial_factorizer)
    op_poly = op_poly.factorize(eqs.mean_field_factorizer)
    print(op_poly**2)
