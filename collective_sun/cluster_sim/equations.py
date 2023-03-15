import collections
import dataclasses
import functools
import itertools
from typing import Callable, Iterator, Optional, Sequence, Union

import numpy as np

import operators as ops


####################################################################################################
# data structures to represent expectation values and polynomials thereof


class ExpectationValueProduct:
    """A product of expectation values of 'MultiBodyOperator's."""

    op_to_exp: collections.defaultdict[ops.MultiBodyOperator, int]

    def __init__(self, *located_ops: ops.MultiBodyOperator) -> None:
        self.op_to_exp = collections.defaultdict(int)
        for located_op in located_ops:
            self.op_to_exp[located_op] += 1

    def __str__(self) -> str:
        if self.is_empty():
            return "<1>"
        return " ".join(f"<{op}>" if exp == 1 else f"<{op}>**{exp}" for op, exp in self)

    def __hash__(self) -> int:
        return hash(tuple(self.op_to_exp.items()))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ExpectationValueProduct) and hash(self) == hash(other)

    def __iter__(self) -> Iterator[tuple[ops.MultiBodyOperator, int]]:
        yield from self.op_to_exp.items()

    def __mul__(
        self, other: Union[ops.MultiBodyOperator, "ExpectationValueProduct"]
    ) -> "ExpectationValueProduct":
        output = ExpectationValueProduct()
        output.op_to_exp = self.op_to_exp.copy()
        if isinstance(other, ops.MultiBodyOperator):
            output.op_to_exp[other] += 1
        else:
            for op, exp in other:
                output.op_to_exp[op] += exp
        return output

    def __rmul__(
        self, other: Union[ops.MultiBodyOperator, "ExpectationValueProduct"]
    ) -> "ExpectationValueProduct":
        return self * other

    def is_empty(self) -> bool:
        return not bool(self.op_to_exp)


@dataclasses.dataclass
class OperatorPolynomial:
    """A polynomial of expectation values of 'ops.MultiBodyOperator's."""

    vec: dict[ExpectationValueProduct, complex]

    def __init__(
        self, initial_vec: Optional[dict[ExpectationValueProduct, complex]] = None
    ) -> None:
        if initial_vec is not None:
            self.vec = collections.defaultdict(complex, initial_vec)
        else:
            self.vec = collections.defaultdict(complex)

    def __str__(self) -> str:
        return "\n".join(f"{scalar} {op}" for op, scalar in self)

    def __iter__(self) -> Iterator[tuple[ExpectationValueProduct, complex]]:
        yield from self.vec.items()

    def __add__(self, other: "OperatorPolynomial") -> "OperatorPolynomial":
        """Add two 'OperatorPolynomial's."""
        output = OperatorPolynomial()
        output.vec = self.vec.copy()
        for term, scalar in other:
            output.vec[term] += scalar
        return output

    def __mul__(self, other: Union[complex, "OperatorPolynomial"]) -> "OperatorPolynomial":
        """Multiply an 'OperatorPolynomial' by a scalar, or by another 'OperatorPolynomial'."""
        output = OperatorPolynomial()
        if isinstance(other, OperatorPolynomial):
            for term_1, scalar_1 in self:
                for term_2, scalar_2 in other:
                    output.vec[term_1 * term_2] += scalar_1 * scalar_2
            return output
        else:
            for term, scalar in self:
                output.vec[term] = other * scalar
            return output

    def __rmul__(self, scalar: complex) -> "OperatorPolynomial":
        return self * scalar

    def __pow__(self, exponent: int) -> "OperatorPolynomial":
        """Raise this 'OperatorPolynomial' to a nonnegative integer power."""
        assert exponent >= 0
        output = OperatorPolynomial()
        if exponent == 0:
            output.vec[ExpectationValueProduct()] = 1
        else:
            output.vec = self.vec.copy()
            for _ in range(1, exponent):
                output = output * self
        return output

    @classmethod
    def from_dense_ops(
        self,
        *terms: ops.DenseMultiBodyOperators | ops.DenseMultiBodyOperator | ops.MultiBodyOperator,
    ) -> "OperatorPolynomial":
        """
        Construct an 'OperatorPolynomial' that represents a sum of the expectation values of
        the given multibody operators.
        """
        output = OperatorPolynomial()

        # loop over the given terms
        for term in terms:
            # if this is a MultiBodyOperator, add its expectation value
            if isinstance(term, ops.MultiBodyOperator):
                product = ExpectationValueProduct(term)
                output.vec[product] += 1
                continue

            if isinstance(term, ops.DenseMultiBodyOperator):
                dense_ops = ops.DenseMultiBodyOperators(term)
            else:  # isinstance(term, ops.DenseMultiBodyOperators)
                dense_ops = term
            dense_ops.simplify()

            # loop over individual 'ops.DenseMultiBodyOperator's in this term
            for dense_op in dense_ops.terms:

                # deal with idendity operators as a special case
                if dense_op.is_identity_op():
                    identity = ExpectationValueProduct()
                    output.vec[identity] += dense_op.scalar * complex(dense_op.tensor)
                    continue

                # collect operators that are fixed to specific lattice sites
                fixed_ops = dense_op.fixed_op.ops

                # loop over all choices of remaning sites to address nontrivially
                available_sites = [
                    site
                    for site in ops.LatticeSite.range(dense_op.num_sites)
                    if site not in dense_op.fixed_sites
                ]
                for addressed_sites in itertools.combinations(
                    available_sites, len(dense_op.dist_ops)
                ):
                    # loop over all assignments of specific operators to specific sites
                    for op_sites in itertools.permutations(addressed_sites):
                        if dense_op.tensor[op_sites]:
                            dist_ops = [
                                ops.SingleBodyOperator(dist_op, site)
                                for dist_op, site in zip(dense_op.dist_ops, op_sites)
                            ]
                            multi_body_op = ops.MultiBodyOperator(*dist_ops, *fixed_ops)
                            product = ExpectationValueProduct(multi_body_op)
                            output.vec[product] += dense_op.scalar * dense_op.tensor[op_sites]

        return output

    @classmethod
    def from_matrix(cls, matrix: np.ndarray, op_mats: Sequence[np.ndarray]) -> "OperatorPolynomial":
        dense_ops = ops.DenseMultiBodyOperators.from_matrix(matrix, op_mats)
        return OperatorPolynomial.from_dense_ops(dense_ops)

    def factorize(self, factorization_rule: "FactorizationRule") -> "OperatorPolynomial":
        """
        Factorize all terms in 'self' according to a given rule for factorizing the expectation
        value of a 'ops.MultiBodyOperator'.
        """
        output = OperatorPolynomial()
        for product_of_expectation_values, scalar in self:

            if product_of_expectation_values.is_empty():
                # this is an identity term
                output.vec[product_of_expectation_values] += scalar
                continue

            factorized_factors = [
                factorization_rule(expectation_value) ** exponent
                for expectation_value, exponent in product_of_expectation_values
            ]
            product_of_factorized_factors = functools.reduce(
                OperatorPolynomial.__mul__,
                factorized_factors,
            )
            output += scalar * product_of_factorized_factors

        return output


FactorizationRule = Callable[[ops.MultiBodyOperator], OperatorPolynomial]


def trivial_factorizer(op: ops.MultiBodyOperator) -> OperatorPolynomial:
    product = ExpectationValueProduct(op)
    return OperatorPolynomial({product: 1})


def mean_field_factorizer(op: ops.MultiBodyOperator) -> OperatorPolynomial:
    factors = [ops.MultiBodyOperator(located_op) for located_op in op]
    product = ExpectationValueProduct(*factors)
    return OperatorPolynomial({product: 1})


####################################################################################################
# methods for building equations of motion


def get_time_derivative(
    op: ops.MultiBodyOperator,
    hamiltonian: ops.DenseMultiBodyOperator,
    structure_factors: np.ndarray,
) -> OperatorPolynomial:
    dense_op = ops.DenseMultiBodyOperator(fixed_op=op, num_sites=hamiltonian.num_sites)
    time_deriv = -1j * ops.commute_dense_ops(dense_op, hamiltonian, structure_factors)
    return OperatorPolynomial.from_dense_ops(time_deriv)


def build_equations_of_motion(
    op_seed: ops.MultiBodyOperator,
    hamiltonian: ops.DenseMultiBodyOperator,
    factorization_rule: FactorizationRule = mean_field_factorizer,
) -> None:
    ...
