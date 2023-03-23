import collections
import dataclasses
import functools
import itertools
import math
import operator
from typing import Callable, Collection, Iterator, Optional, Sequence, TypeVar, Union

import numpy as np
import sparse

import operators as ops

GenericType = TypeVar("GenericType")

####################################################################################################
# data structures to represent expectation values and polynomials thereof


class ExpectationValueProduct:
    """A product of expectation values of 'MultiBodyOperator's."""

    op_to_exp: collections.defaultdict[ops.MultiBodyOperator, int]

    def __init__(self, *located_ops: ops.MultiBodyOperator) -> None:
        self.op_to_exp = collections.defaultdict(int)
        for located_op in located_ops:
            if not located_op.is_identity_op():
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

    def factors(self) -> tuple[ops.MultiBodyOperator, ...]:
        return functools.reduce(operator.add, [(op,) * exp for op, exp in self], ())

    def prime_factors(self) -> tuple[ops.MultiBodyOperator, ...]:
        return tuple(op for op, _ in self)

    def num_factors(self) -> int:
        return sum(self.op_to_exp.values(), start=0)

    def num_prime_factors(self) -> int:
        return len(self.op_to_exp)

    def is_empty(self) -> bool:
        return not bool(self.op_to_exp)


@dataclasses.dataclass
class OperatorPolynomial:
    """A polynomial of expectation values of 'ops.MultiBodyOperator's."""

    vec: dict[ExpectationValueProduct, complex]

    def __init__(
        self,
        initializer: Optional[
            dict[ExpectationValueProduct, complex]
            | ExpectationValueProduct
            | ops.MultiBodyOperator
            | ops.SingleBodyOperator
        ] = None,
    ) -> None:
        if isinstance(initializer, ops.SingleBodyOperator):
            initializer = ops.MultiBodyOperator(initializer)
        if isinstance(initializer, ops.MultiBodyOperator):
            initializer = ExpectationValueProduct(initializer)
        if isinstance(initializer, ExpectationValueProduct):
            initializer = {initializer: 1}
        if initializer is not None:
            self.vec = collections.defaultdict(complex, initializer)
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
            if output.vec[term] == 0:
                output.vec.pop(term)
        return output

    def __sub__(self, other: "OperatorPolynomial") -> "OperatorPolynomial":
        """Subtract one 'OperatorPolynomial' from another."""
        output = OperatorPolynomial()
        output.vec = self.vec.copy()
        for term, scalar in other:
            output.vec[term] -= scalar
            if output.vec[term] == 0:
                output.vec.pop(term)
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
    def from_ops(
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
    def from_matrix(
        cls, matrix: np.ndarray, op_mats: Sequence[np.ndarray], cutoff: float = 0
    ) -> "OperatorPolynomial":
        dense_ops = ops.DenseMultiBodyOperators.from_matrix(matrix, op_mats, cutoff)
        return OperatorPolynomial.from_ops(dense_ops)

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

    def to_array(
        self,
        index_map: dict[ExpectationValueProduct, int] | dict[ops.MultiBodyOperator, int],
    ) -> np.ndarray:
        """Convert this polynomial into an array."""
        # insure that our index_map takes ExpectationValueProduct --> int
        index_map = {
            key if isinstance(key, ExpectationValueProduct) else ExpectationValueProduct(key): val
            for key, val in index_map.items()
        }
        # construct the array
        output = np.zeros(len(index_map), dtype=complex)
        for product, value in self:
            output[index_map[product]] = value
        return output

    @classmethod
    def from_product_state(
        cls, product_state: np.ndarray, op_mats: Sequence[np.ndarray]
    ) -> "OperatorPolynomial":
        """Construct local expectation values from a product state."""
        norm = np.prod(np.einsum("sj,sj->s", product_state.conj(), product_state))
        assert np.isclose(norm, 1)

        # compute all local expectation values
        output = OperatorPolynomial(ops.MultiBodyOperator())
        num_sites, _ = product_state.shape
        for site in ops.LatticeSite.range(num_sites):
            for op_idx, op_mat in enumerate(op_mats[1:], start=1):
                local_op = ops.SingleBodyOperator(op_idx, site)
                op = ExpectationValueProduct(ops.MultiBodyOperator(local_op))
                val = product_state[site, :].conj() @ op_mat @ product_state[site, :]
                if np.isclose(val.imag, 0):
                    val = val.real
                if np.isclose(val.real, 0):
                    val = val.imag
                if val:
                    output.vec[op] = val

        return output


####################################################################################################
# methods for factorizing expectation values


FactorizationRule = Callable[[ops.MultiBodyOperator], OperatorPolynomial]


def trivial_factorizer(op: ops.MultiBodyOperator) -> OperatorPolynomial:
    return OperatorPolynomial.from_ops(op)


def mean_field_factorizer(op: ops.MultiBodyOperator) -> OperatorPolynomial:
    factors = [ops.MultiBodyOperator(located_op) for located_op in op]
    product = ExpectationValueProduct(*factors)
    return OperatorPolynomial(product)


def get_cumulant_factorizer(
    keep: Callable[[ops.MultiBodyOperator], bool] = lambda _: False
) -> FactorizationRule:
    """Construct a factorization rule obtained by setting cumulants to zero."""

    def cumulant_factorizer(op: ops.MultiBodyOperator) -> OperatorPolynomial:
        """Factorize a multi-body operator by setting the cumulant of its local factors to zero."""
        if op.locality <= 1 or keep(op):
            return OperatorPolynomial(op)
        factorized_op = OperatorPolynomial(op) - get_cumulant(*op.ops)
        return factorized_op.factorize(cumulant_factorizer)

    return cumulant_factorizer


def get_cumulant(*local_ops: ops.SingleBodyOperator) -> OperatorPolynomial:
    """Return the cumulant of a collection of local operators."""
    assert len(set(op.site for op in local_ops)) == len(local_ops)
    output = OperatorPolynomial()
    for partition in partitions(local_ops):
        sign = (-1) ** (len(partition) - 1)
        prefactor = math.factorial(len(partition) - 1)
        factors = [ops.MultiBodyOperator(*part) for part in partition]
        product = ExpectationValueProduct(*factors)
        output += sign * prefactor * OperatorPolynomial(product)
    return output


def partitions(collection: Collection[GenericType]) -> Iterator[set[frozenset[GenericType]]]:
    """Iterate over all partitions of a collection of unique items.
    Algorithm adapted from https://stackoverflow.com/a/30134039.
    """
    if len(collection) == 1:
        yield {frozenset(collection)}

    else:
        # convert the collection into a set, and pick out one ("first") item
        items_set = set(collection)
        first_item = items_set.pop()

        # iterate over all partitions of the remaining items in the set
        for sub_partition in partitions(items_set):

            # put the `first_item` in its own subset
            yield {frozenset([first_item])} | sub_partition

            # insert the `first_item` into each of the `sub_partition`s subsets
            for part in sub_partition:
                yield (sub_partition - {part}) | {part | {first_item}}


####################################################################################################
# methods for building equations of motion


def get_time_derivative(
    op: ops.MultiBodyOperator,
    hamiltonian: ops.DenseMultiBodyOperators | ops.DenseMultiBodyOperator,
    structure_factors: np.ndarray,
) -> OperatorPolynomial:
    if isinstance(hamiltonian, ops.DenseMultiBodyOperator):
        hamiltonian = ops.DenseMultiBodyOperators(hamiltonian)
    dense_op = ops.DenseMultiBodyOperator(fixed_op=op, num_sites=hamiltonian.num_sites)
    time_deriv = -1j * ops.commute_dense_ops(dense_op, hamiltonian, structure_factors)
    return OperatorPolynomial.from_ops(time_deriv)


def build_equations_of_motion(
    *op_seeds: ops.MultiBodyOperator | ExpectationValueProduct,
    hamiltonian: ops.DenseMultiBodyOperators | ops.DenseMultiBodyOperator,
    structure_factors: np.ndarray,
    factorization_rule: FactorizationRule = trivial_factorizer,
    show_progress: bool = False,
) -> tuple[dict[ops.MultiBodyOperator, int], tuple[sparse.SparseArray, ...]]:
    if isinstance(hamiltonian, ops.DenseMultiBodyOperator):
        hamiltonian = ops.DenseMultiBodyOperators(hamiltonian)

    # construct an initial set of ops to differentiate
    ops_to_differentiate = set()
    for seed in op_seeds:
        if isinstance(seed, ops.MultiBodyOperator):
            ops_to_differentiate.add(seed)
        elif isinstance(seed, ExpectationValueProduct):
            for op in seed.prime_factors():
                ops_to_differentiate.add(op)
            if seed.is_empty():
                ops_to_differentiate.add(ops.MultiBodyOperator())

    # compute all time derivatives
    time_derivs: dict[ops.MultiBodyOperator, OperatorPolynomial] = {}
    while ops_to_differentiate:
        if show_progress:
            print(len(ops_to_differentiate))
        op = ops_to_differentiate.pop()
        time_deriv = get_time_derivative(op, hamiltonian, structure_factors)
        time_deriv = time_deriv.factorize(factorization_rule)
        time_derivs[op] = time_deriv

        # update the set of ops_to_differentiate
        for expectation_values, _ in time_deriv:
            for factor, __ in expectation_values:
                if factor not in time_derivs:
                    ops_to_differentiate.add(factor)

    # assign an integer index to each operator
    dim = len(time_derivs)
    op_to_index = {op: ii for ii, op in enumerate(sorted(time_derivs.keys()))}

    # construct time derivative tensors
    time_deriv_tensors = {}
    for op, time_deriv in time_derivs.items():
        for term, val in time_deriv:
            factors = term.factors()
            num_factors = len(factors)
            if num_factors not in time_deriv_tensors:
                shape = (dim,) * (num_factors + 1)
                time_deriv_tensors[num_factors] = sparse.DOK(shape, dtype=complex)
            time_deriv_tensors[num_factors]
            indices = tuple(op_to_index[factor] for factor in [op, *factors])
            time_deriv_tensors[num_factors][indices] = val

    return op_to_index, tuple(tensor.to_coo() for tensor in time_deriv_tensors.values())


def time_deriv(
    _: float, op_vec: np.ndarray, time_deriv_tensors: tuple[sparse.SparseArray | np.ndarray, ...]
) -> np.ndarray:
    output = sum(_single_tensor_time_deriv(op_vec, tensor) for tensor in time_deriv_tensors)
    if not isinstance(output, np.ndarray):
        return np.zeros_like(op_vec)
    return output


def _single_tensor_time_deriv(
    op_vec: np.ndarray, time_deriv_tensor: sparse.SparseArray | np.ndarray
) -> np.ndarray:
    order = time_deriv_tensor.ndim - 1
    vec_indices = "abcdefghijklmnopqrstuvwxyz"[:order]
    indices = f"Z{vec_indices}," + ",".join(tuple(vec_indices)) + "->Z"
    op_vecs = [op_vec] * order
    if isinstance(time_deriv_tensor, sparse.SparseArray):
        return sparse.einsum(indices, time_deriv_tensor, *op_vecs).todense()
    return np.einsum(indices, time_deriv_tensor, *op_vecs)
