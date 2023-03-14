import pytest

import operators as ops
import equations as eqs


qubit_op_mats = ops.get_qubit_op_mats()


@pytest.mark.parametrize("num_sites", [3])
def test_op_poly(num_sites: int) -> None:
    op_mat = ops.get_random_op(2**num_sites)
    op_sum = ops.DenseMultiBodyOperators.from_matrix(op_mat, qubit_op_mats)
    op_poly = eqs.OperatorPolynomial.from_multi_body_ops(op_sum)

    def factorization_rule(located_ops: ops.MultiBodyOperator) -> eqs.OperatorPolynomial:
        output = eqs.OperatorPolynomial()
        factors = [ops.MultiBodyOperator(located_op) for located_op in located_ops]
        product = eqs.ExpectationValueProduct(*factors)
        output.vec[product] = 1
        return output

    op_poly = op_poly.factorize(factorization_rule)
    print(op_poly**2)
