#!/usr/bin/env python3
import collections
import dataclasses
import functools
import itertools
from typing import Callable, Iterator, Optional, Sequence, TypeVar, Union

import numpy as np


def tensor_product(tensor_a: np.ndarray, tensor_b: np.ndarray) -> np.ndarray:
    return np.tensordot(tensor_a, tensor_b, axes=0)


####################################################################################################
# methods for computing structure factors


def trace_inner_product(op_a: np.ndarray, op_b: np.ndarray) -> complex:
    """Inner product: <A, B> = Tr[A^dag B] / dim."""
    return (op_a.conj() * op_b).sum() / op_a.shape[0]


def multiply_mats(op_a: np.ndarray, op_b: np.ndarray) -> np.ndarray:
    return op_a @ op_b


def commute_mats(op_a: np.ndarray, op_b: np.ndarray) -> np.ndarray:
    return op_a @ op_b - op_b @ op_a


def get_structure_factors(
    op_mat_I: np.ndarray,
    *other_mats: np.ndarray,
    binary_op: Callable[[np.ndarray, np.ndarray], np.ndarray] = multiply_mats,
    inner_product: Callable[[np.ndarray, np.ndarray], complex] = trace_inner_product,
) -> np.ndarray:
    """
    Compute the structure factors f_{ABC} for the operator algebra associated with the given
    matrices:
        A @ B = sum_C f_{ABC} C
    """
    dim = op_mat_I.shape[0]
    op_mats = [op_mat_I, *other_mats]
    assert np.array_equal(op_mat_I, np.eye(dim))
    assert len(op_mats) == dim**2

    structure_factors = np.empty((len(op_mats),) * 3, dtype=complex)
    for idx_a, mat_a in enumerate(other_mats, start=1):
        for idx_b, mat_b in enumerate(other_mats, start=1):
            mat_comm = binary_op(mat_a, mat_b)
            mat_comm_vec = [inner_product(op_mat, mat_comm) for op_mat in op_mats]
            structure_factors[idx_a, idx_b, :] = mat_comm_vec
    return structure_factors


####################################################################################################
# classes for "typed" integers (or indices)


Self = TypeVar("Self", bound="TypedInteger")


@dataclasses.dataclass
class TypedInteger:
    index: int

    def __str__(self) -> str:
        return str(self.index)

    def __hash__(self) -> int:
        return hash(str(self))

    def __index__(self) -> int:
        return self.index

    def __lt__(self, other: "TypedInteger") -> bool:
        return self.index < other.index

    @classmethod
    def range(cls: type[Self], *args: int) -> Iterator[Self]:
        yield from (cls(index) for index in range(*args))


class LatticeSite(TypedInteger):
    """A "site", indexing a physical location."""


####################################################################################################
# data structures for representing operators


class AbstractSingleBodyOperator(TypedInteger):
    """An "abstract" single-body, not associated with any particular site."""

    def __str__(self) -> str:
        return f"O_{self.index}"

    def to_matrix(self, op_mats: Sequence[np.ndarray]) -> np.ndarray:
        return op_mats[self.index]

    def is_identity_op(self) -> bool:
        return self.index == 0


@dataclasses.dataclass
class SingleBodyOperator:
    """A single-body operator located at a specific site."""

    op: AbstractSingleBodyOperator
    site: LatticeSite

    def __str__(self) -> str:
        return f"{self.op}({self.site})"

    def __hash__(self) -> int:
        return hash(str(self))

    def is_identity_op(self) -> bool:
        return self.op.is_identity_op()


@dataclasses.dataclass
class MultiBodyOperator:
    """A product of (non-identity) single-body operators located at specific sites."""

    ops: frozenset[SingleBodyOperator]

    def __init__(self, *located_ops: SingleBodyOperator) -> None:
        self.ops = frozenset(op for op in located_ops if not op.is_identity_op())

    def __str__(self) -> str:
        return " ".join(str(op) for op in self)

    def __hash__(self) -> int:
        return hash(str(self))

    def __iter__(self) -> Iterator[SingleBodyOperator]:
        yield from self.ops

    def __bool__(self) -> bool:
        return bool(self.ops)

    @property
    def locality(self) -> int:
        return len(self.ops)

    def is_identity_op(self) -> bool:
        return len(self.ops) == 0


@dataclasses.dataclass
class DenseMultiBodyOperator:
    scalar: complex
    tensor: np.ndarray
    local_ops: tuple[AbstractSingleBodyOperator, ...]
    fixed_ops: MultiBodyOperator

    def __init__(
        self,
        tensor: np.ndarray,
        *local_ops: AbstractSingleBodyOperator,
        scalar: complex = 1,
        fixed_ops: MultiBodyOperator = MultiBodyOperator(),
        num_sites: Optional[int] = None,
        simplify: bool = True,
    ) -> None:
        if tensor.ndim != len(local_ops):
            raise ValueError("tensor dimension does not match operator locality")
        self.scalar = scalar
        self.tensor = tensor
        self.local_ops = local_ops
        self.fixed_ops = fixed_ops

        if num_sites is not None:
            self._num_sites = num_sites
        elif tensor.ndim:
            self._num_sites = tensor.shape[0]
        else:
            raise ValueError("not enough data provided to determine operator locality")
        if tensor.ndim:
            assert all(dim == self._num_sites for dim in tensor.shape)

        self._locality = self.fixed_ops.locality + len(
            [op for op in local_ops if not op.is_identity_op()]
        )

        if simplify:
            self.simplify()

    def __str__(self) -> str:
        fixed_op_str = "{self.fixed_op}; " if self.fixed_ops else ""
        local_op_str = " ".join(str(op) for op in self.local_ops)
        return f"D({fixed_op_str}{local_op_str})"

    def __mul__(self, scalar: complex) -> "DenseMultiBodyOperator":
        new_scalar = self.scalar * scalar
        return DenseMultiBodyOperator(self.tensor, *self.local_ops, scalar=new_scalar)

    def __rmul__(self, scalar: complex) -> "DenseMultiBodyOperator":
        return self * scalar

    def __bool__(self) -> bool:
        """Return 'True' if and only if this operator is nonzero."""
        return bool(self.scalar) and bool(self.tensor.any())

    @property
    def num_sites(self) -> int:
        return self._num_sites

    @property
    def locality(self) -> int:
        return self._locality

    def is_identity_op(self) -> bool:
        return self._locality == 0

    def simplify(self) -> None:
        """Simplify 'self' by removing any identity operators from 'self.local_ops'."""
        # identity all nontrivial (non-identity) local operators
        non_identity_data = [
            (idx, local_op)
            for idx, local_op in enumerate(self.local_ops)
            if not local_op.is_identity_op()
        ]
        if len(non_identity_data) != self.tensor.ndim:
            # trace over the indices in the tensor associated with identity operators
            non_identity_indices, non_identity_ops = zip(*non_identity_data)
            self.tensor = np.einsum(self.tensor, range(self.tensor.ndim), non_identity_indices)
            self.local_ops = non_identity_ops

    def in_canonical_form(self) -> "DenseMultiBodyOperator":
        """Rearrange data to sort 'self.local_ops' by increasing index."""
        argsort = np.argsort([local_op.index for local_op in self.local_ops])
        tensor = np.transpose(self.tensor, argsort)
        local_ops = [self.local_ops[idx] for idx in argsort]
        return DenseMultiBodyOperator(tensor, *local_ops, scalar=self.scalar)

    def to_tensor(self, op_mats: Sequence[np.ndarray]) -> np.ndarray:
        """
        Return a tensor representation of 'self'.

        Consider an operator that can be factorized into a single local operator for each lattice
        site.  Such an operator can be represented by a tensor of the form
          't = otimes_{site s} vect(O_s)',
        where
        - 'otimes_{site s}' denotes a tensor product over all sites 's'
        - 'O_s' is the operator addressing site 's', and
        - 'vect(O)' is a vectorized version of the matrix representing 'O'.

        Given this tensor representation of "product operators", we can represent 'self' with the
        tensor
            'T = sum_t c_t t',
        where 'c_t' is the coefficient for 't' in 'self'.
        """
        spin_dim = int(np.round(np.sqrt(len(op_mats))))
        if self.is_identity_op():
            return self.tensor * self.scalar * _get_iden_op_tensor(spin_dim, self.num_sites)

        op_mat_I = np.eye(spin_dim)
        assert np.array_equal(op_mats[0], op_mat_I)

        # construct a tensor for all sites addressed by the identity
        self.simplify()
        iden_op_tensor = _get_iden_op_tensor(spin_dim, self.num_sites - self.locality)

        # vectorize all nontrivial operators in 'self', and identify fixed/nonfixed sites
        local_op_vecs = [local_op.to_matrix(op_mats).ravel() for local_op in self.local_ops]
        if self.fixed_ops:
            fixed_op_vecs, fixed_op_sites = zip(
                *[(op.op.to_matrix(op_mats).ravel(), op.site) for op in self.fixed_ops]
            )
        else:
            fixed_op_vecs, fixed_op_sites = [], ()
        non_fixed_op_sites = [idx for idx in range(self.num_sites) if idx not in fixed_op_sites]

        # take a tensor product of vectorized operators for all sites
        base_tensor = self.scalar * functools.reduce(
            tensor_product, fixed_op_vecs + local_op_vecs + [iden_op_tensor]
        )

        # sum over all choices of sites to address nontrivially
        op_tensors = (
            self.tensor[local_op_sites]
            * np.moveaxis(base_tensor, range(self.locality), fixed_op_sites + local_op_sites)
            for non_fixed_sites in itertools.combinations(non_fixed_op_sites, self.tensor.ndim)
            for local_op_sites in itertools.permutations(non_fixed_sites)
            if self.tensor[local_op_sites]
        )
        return sum(op_tensors)


@functools.cache
def _get_iden_op_tensor(spin_dim: int, num_sites: int) -> np.ndarray:
    """Return a tensor product of flattened identity matrices."""
    iden_tensors = [np.eye(spin_dim).ravel()] * num_sites
    return functools.reduce(tensor_product, iden_tensors, np.array(1))


@dataclasses.dataclass
class DenseMultiBodyOperators:
    ops: list[DenseMultiBodyOperator]

    def __init__(
        self,
        *terms: Union[DenseMultiBodyOperator, "DenseMultiBodyOperators"],
        simplify: bool = True,
    ) -> None:
        if terms:
            assert len(set(term.num_sites for term in terms)) == 1
        self.ops = [term for term in terms if isinstance(term, DenseMultiBodyOperator)] + [
            op for term in terms if isinstance(term, DenseMultiBodyOperators) for op in term.ops
        ]
        if simplify:
            self.simplify()

    def __mul__(self, scalar: complex) -> "DenseMultiBodyOperators":
        new_ops = [scalar * op for op in self.ops]
        return DenseMultiBodyOperators(*new_ops)

    def __rmul__(self, scalar: complex) -> "DenseMultiBodyOperators":
        return self * scalar

    def __add__(
        self, other: Union[DenseMultiBodyOperator, "DenseMultiBodyOperators"]
    ) -> "DenseMultiBodyOperators":
        assert self.num_sites == other.num_sites or self.num_sites == 0 or other.num_sites == 0
        return DenseMultiBodyOperators(
            *self.ops, *DenseMultiBodyOperators(other).ops, simplify=False
        )

    def __iadd__(
        self, other: Union[DenseMultiBodyOperator, "DenseMultiBodyOperators"]
    ) -> "DenseMultiBodyOperators":
        self = self + other
        return self

    def __iter__(self) -> Iterator[DenseMultiBodyOperator]:
        yield from self.ops

    @property
    def num_sites(self) -> int:
        if not self.ops:
            return 0
        return self.ops[0].num_sites

    def simplify(self) -> None:
        """
        Simplify 'self' by removing trivial (zero) terms, and combining terms that have the same
        operator content.
        """
        # remove trivial (zero) terms
        self.ops = [op for op in self.ops if op]

        # combine terms that are the same up to a permutation of local operators
        local_op_counts = [collections.Counter(op.local_ops) for op in self.ops]
        for jj in reversed(range(1, len(local_op_counts))):
            for ii in range(jj):
                # if terms ii and jj have the same operator content, merge term jj into ii
                if local_op_counts[ii] == local_op_counts[jj]:
                    op_ii = self.ops[ii].in_canonical_form()
                    op_jj = self.ops[jj].in_canonical_form()
                    local_ops = op_ii.local_ops
                    if np.array_equal(op_ii.tensor, op_jj.tensor):
                        tensor = op_ii.tensor
                        scalar = op_ii.scalar + op_jj.scalar
                    else:
                        tensor = op_ii.tensor + op_jj.scalar / op_ii.scalar * op_jj.tensor
                        scalar = op_ii.scalar
                    new_op = DenseMultiBodyOperator(
                        tensor, *local_ops, scalar=scalar, num_sites=self.num_sites
                    )
                    self.ops[ii] = new_op
                    del self.ops[jj]
                    break

    @classmethod
    def from_matrix(
        cls, matrix: np.ndarray, op_mats: Sequence[np.ndarray]
    ) -> "DenseMultiBodyOperators":
        """
        Construct a 'DenseMultiBodyOperators' object from the matrix representation of an operator
        on a Hibert space.
        """
        spin_dim = int(np.round(np.sqrt(len(op_mats))))
        num_sites = int(np.round(np.log(matrix.size) / np.log(spin_dim))) // 2

        op_mat_I = np.eye(spin_dim)
        assert np.array_equal(op_mats[0], op_mat_I)

        output = DenseMultiBodyOperators()

        # add an identity term, which gets special treatment in the 'DenseMultiBodyOperator' class
        identity_coefficient = trace_inner_product(np.eye(spin_dim**num_sites), matrix)
        if identity_coefficient:
            output += DenseMultiBodyOperator(
                np.array(1), scalar=identity_coefficient, num_sites=num_sites
            )

        # loop over all numbers of sites that may be addressed nontrivially
        for locality in range(1, num_sites + 1):
            # loop over all choices of specific sites that are addressed nontrivially
            for non_iden_sites in itertools.combinations(range(num_sites), locality):

                # initialize a coefficient tensor to populate with nonzero values
                tensor = np.zeros((num_sites,) * locality, dtype=complex)

                # loop over all choices of nontrivial operators at the chosen sites
                for local_ops in itertools.product(
                    AbstractSingleBodyOperator.range(1, len(op_mats)), repeat=locality
                ):
                    # compute the coefficient for this choice of local operators
                    matrices = [op_mat_I] * num_sites
                    for idx, local_op in zip(non_iden_sites, local_ops):
                        matrices[idx] = local_op.to_matrix(op_mats)
                    term_matrix = functools.reduce(np.kron, matrices)
                    coefficient = trace_inner_product(term_matrix, matrix)

                    # add this term to the output
                    if coefficient:
                        tensor[non_iden_sites] = coefficient
                        output += DenseMultiBodyOperator(
                            tensor.copy(), *local_ops, num_sites=num_sites
                        )

        output.simplify()
        return output

    def to_matrix(self, op_mats: Sequence[np.ndarray]) -> np.ndarray:
        """Return the matrix representation of 'self'."""
        spin_dim = int(np.round(np.sqrt(len(op_mats))))
        num_sites = self.num_sites
        output_matrix_shape = (spin_dim**num_sites,) * 2

        op_tensor = sum((op.to_tensor(op_mats) for op in self.ops))
        if not isinstance(op_tensor, np.ndarray):
            return np.zeros(output_matrix_shape)

        op_tensor.shape = (spin_dim,) * (2 * num_sites)
        op_tensor = np.moveaxis(
            op_tensor, range(1, 2 * num_sites, 2), range(num_sites, 2 * num_sites)
        )
        return op_tensor.reshape(output_matrix_shape)


####################################################################################################
# data structures to represent expectation values


class ExpectationValueProduct:
    """A product of expectation values of 'MultiBodyOperator's."""

    _op_to_exp: collections.defaultdict[MultiBodyOperator, int]

    def __init__(self, *located_ops: MultiBodyOperator) -> None:
        self._op_to_exp = collections.defaultdict(int)
        for located_op in located_ops:
            self._op_to_exp[located_op] += 1

    def __str__(self) -> str:
        if self.is_empty():
            return "<1>"
        return " ".join(f"<{op}>" if exp == 1 else f"<{op}>**{exp}" for op, exp in self)

    def __hash__(self) -> int:
        return hash(tuple(self._op_to_exp.items()))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ExpectationValueProduct) and hash(self) == hash(other)

    def __iter__(self) -> Iterator[tuple[MultiBodyOperator, int]]:
        yield from self._op_to_exp.items()

    def __mul__(
        self, other: Union[MultiBodyOperator, "ExpectationValueProduct"]
    ) -> "ExpectationValueProduct":
        output = ExpectationValueProduct()
        output._op_to_exp = self._op_to_exp.copy()
        if isinstance(other, MultiBodyOperator):
            output._op_to_exp[other] += 1
        else:
            for op, exp in other:
                output._op_to_exp[op] += exp
        return output

    def __rmul__(
        self, other: Union[MultiBodyOperator, "ExpectationValueProduct"]
    ) -> "ExpectationValueProduct":
        return self * other

    def is_empty(self) -> bool:
        return not bool(self._op_to_exp)


@dataclasses.dataclass
class OperatorPolynomial:
    """A polynomial of expectation values of 'MultiBodyOperator's."""

    vec: dict[ExpectationValueProduct, complex]

    def __init__(self) -> None:
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
        """Raise this 'OperatorPolynomial' to a positive integer power."""
        assert exponent > 0
        output = OperatorPolynomial()
        output.vec = self.vec.copy()
        for _ in range(1, exponent):
            output = output * self
        return output

    @classmethod
    def from_multi_body_ops(
        self, *terms: DenseMultiBodyOperators | DenseMultiBodyOperator
    ) -> "OperatorPolynomial":
        """
        Construct an 'OperatorPolynomial' that represents a sum of the expectation values of
        the given multibody operators.
        """
        output = OperatorPolynomial()

        # loop over the given terms
        for dense_ops in terms:
            if isinstance(dense_ops, DenseMultiBodyOperator):
                dense_ops = DenseMultiBodyOperators(dense_ops)
            dense_ops.simplify()

            # loop over individual 'DenseMultiBodyOperator's in this term
            for dense_op in dense_ops.ops:

                # deal with idendity operators as a special case
                if dense_op.is_identity_op():
                    identity = ExpectationValueProduct()
                    output.vec[identity] += dense_op.scalar * complex(dense_op.tensor)
                    continue

                # loop over all choices of sites to address nontrivially
                for addressed_sites in itertools.combinations(
                    LatticeSite.range(dense_op.num_sites), dense_op.locality
                ):
                    # loop over all assignments of specific operators to specific sites
                    for op_sites in itertools.permutations(addressed_sites):
                        if dense_op.tensor[op_sites]:
                            single_body_ops = [
                                SingleBodyOperator(local_op, site)
                                for local_op, site in zip(dense_op.local_ops, op_sites)
                            ]
                            multi_body_op = MultiBodyOperator(*single_body_ops)
                            term = ExpectationValueProduct(multi_body_op)
                            output.vec[term] += dense_op.scalar * dense_op.tensor[op_sites]

        return output

    def factorize(
        self, factorization_rule: Callable[[MultiBodyOperator], "OperatorPolynomial"]
    ) -> "OperatorPolynomial":
        """
        Factorize all terms in 'self' according to a given rule for factorizing the expectation
        value of a 'MultiBodyOperator'.
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


####################################################################################################
# methods for commuting operators


def commute_dense_ops(
    op_a: DenseMultiBodyOperators | DenseMultiBodyOperator,
    op_b: DenseMultiBodyOperators | DenseMultiBodyOperator,
    structure_factors: np.ndarray,
    simplify: bool = True,
) -> DenseMultiBodyOperators:
    """Compute the commutator of two dense multibody operators."""
    op_a = DenseMultiBodyOperators(op_a)
    op_b = DenseMultiBodyOperators(op_b)

    commutator_factor_func = _get_multibody_commutator_factor_func(structure_factors)

    output = DenseMultiBodyOperators()
    for term_a, term_b in itertools.product(op_a.ops, op_b.ops):
        output += _commute_dense_op_terms(term_a, term_b, commutator_factor_func)
    if simplify:
        output.simplify()
    return output


CommutatorFactorFuncType = Callable[
    [tuple[AbstractSingleBodyOperator, ...], tuple[AbstractSingleBodyOperator, ...]],
    np.ndarray,
]


def _get_multibody_commutator_factor_func(
    structure_factors: np.ndarray,
) -> CommutatorFactorFuncType:
    """
    Return a function that computes the coefficients in the commutator of two multibody operators.
    """

    @functools.cache
    def commutator_factor_func(
        overlap_ops_a: tuple[AbstractSingleBodyOperator, ...],
        overlap_ops_b: tuple[AbstractSingleBodyOperator, ...],
    ) -> np.ndarray:
        """Get the coefficients in the commutator of two products of local operators."""
        factors_ab = functools.reduce(
            tensor_product,
            [structure_factors[aa, bb] for aa, bb in zip(overlap_ops_a, overlap_ops_b)],
        )
        factors_ba = functools.reduce(
            tensor_product,
            [structure_factors[bb, aa] for aa, bb in zip(overlap_ops_a, overlap_ops_b)],
        )
        return factors_ab - factors_ba

    return commutator_factor_func


def _commute_dense_op_terms(
    op_a: DenseMultiBodyOperator,
    op_b: DenseMultiBodyOperator,
    commutator_factor_func: CommutatorFactorFuncType,
) -> DenseMultiBodyOperators:
    """Compute the commutator of two 'DenseMultiBodyOperator's."""
    assert op_a.num_sites == op_b.num_sites

    if op_a.is_identity_op() or op_b.is_identity_op():
        return DenseMultiBodyOperators()

    output = DenseMultiBodyOperators()
    overlap_scalar = op_a.scalar * op_b.scalar
    min_overlaps = max(1, op_a.locality + op_b.locality - op_a.num_sites)
    max_overlaps = min(op_a.locality, op_b.locality)
    for num_overlaps in range(min_overlaps, max_overlaps + 1):

        # loop over choices for which operators in 'op_a' to overlap with 'op_b'
        overlap_index_set_a = itertools.combinations(range(op_a.locality), num_overlaps)
        for overlap_indices_a in overlap_index_set_a:
            non_overlap_indices_a = tuple(
                idx for idx in range(op_a.locality) if idx not in overlap_indices_a
            )

            # local operators in 'op_a' that will overlap in these terms
            overlap_ops_a = tuple(op_a.local_ops[idx] for idx in overlap_indices_a)
            non_overlap_ops_a = tuple(op_a.local_ops[idx] for idx in non_overlap_indices_a)

            # loop over choices for operators in 'op_b' to overlap with 'overlap_ops_a'
            overlap_index_set_b = (
                permuted_indices
                for indices in itertools.combinations(range(op_b.locality), num_overlaps)
                for permuted_indices in itertools.permutations(indices)
            )
            for overlap_indices_b in overlap_index_set_b:
                non_overlap_indices_b = [
                    idx for idx in range(op_b.locality) if idx not in overlap_indices_b
                ]

                # construct the tensor of spacial (site) coefficients all terms with these overlaps
                overlap_tensor = _get_overlap_tensor(
                    op_a.tensor,
                    op_b.tensor,
                    overlap_indices_a,
                    overlap_indices_b,
                )
                if not overlap_tensor.any():
                    continue

                # construct list of the local operators that will overlap 'overlap_ops_a'
                overlap_ops_b = tuple(op_b.local_ops[idx] for idx in overlap_indices_b)
                non_overlap_ops_b = tuple(op_b.local_ops[idx] for idx in non_overlap_indices_b)

                # identify coefficients for terms in the commutator of the overlapping ops
                commutator_factors = commutator_factor_func(overlap_ops_a, overlap_ops_b)

                # add all terms in the commutator with these overlaps
                output += _commute_local_ops(
                    commutator_factors,
                    non_overlap_ops_a + non_overlap_ops_b,
                    overlap_tensor,
                    overlap_scalar,
                )

    return output


def _get_overlap_tensor(
    tensor_a: np.ndarray,
    tensor_b: np.ndarray,
    overlap_indices_a: tuple[int, ...],
    overlap_indices_b: tuple[int, ...],
) -> np.ndarray:
    """
    Construct the tensor obtained by enforcing that indices 'overlap_indices_a' of 'tensor_a' are
    equal to indices 'overlap_indices_b' of 'tensor_b'.

    Note that the overlapped indices will be the first indices of the output, while all other
    indices preserve their order.
    """
    indices_a = range(tensor_a.ndim)
    indices_b = list(range(tensor_a.ndim, tensor_a.ndim + tensor_b.ndim))
    for idx_a, idx_b in zip(overlap_indices_a, overlap_indices_b):
        indices_b[idx_b] = idx_a
    non_overlap_indices_a = tuple(idx for idx in indices_a if idx not in overlap_indices_a)
    non_overlap_indices_b = tuple(idx for idx in indices_b if idx >= tensor_a.ndim)
    indices_final = overlap_indices_a + non_overlap_indices_a + non_overlap_indices_b
    return np.einsum(tensor_a, indices_a, tensor_b, indices_b, indices_final)


def _commute_local_ops(
    commutator_factors: np.ndarray,
    additional_ops: tuple[AbstractSingleBodyOperator, ...],
    tensor: np.ndarray,
    scalar: complex = 1,
) -> DenseMultiBodyOperators:
    """Commute overlapping local operators."""
    output = DenseMultiBodyOperators()

    # add nonzero terms to the output
    for overlap_ops_by_index in np.argwhere(commutator_factors):
        factor = commutator_factors[tuple(overlap_ops_by_index)]
        overlap_ops = [AbstractSingleBodyOperator(cc) for cc in overlap_ops_by_index]
        output += DenseMultiBodyOperator(
            tensor, *overlap_ops, *additional_ops, scalar=scalar * factor
        )

    return output


####################################################################################################

op_I = AbstractSingleBodyOperator(0)
op_Z = AbstractSingleBodyOperator(1)
op_X = AbstractSingleBodyOperator(2)
op_Y = AbstractSingleBodyOperator(3)
ops = [op_I, op_Z, op_X, op_Y]

op_mat_I = np.eye(2)
op_mat_Z = np.array([[1, 0], [0, -1]])
op_mat_X = np.array([[0, 1], [1, 0]])
op_mat_Y = -1j * op_mat_Z @ op_mat_X
op_mats = [op_mat_I, op_mat_Z, op_mat_X, op_mat_Y]

structure_factors = get_structure_factors(*op_mats)


def get_random_op(num_sites: int) -> np.ndarray:
    real = np.random.random((2**num_sites,) * 2)
    imag = np.random.random((2**num_sites,) * 2)
    return real + 1j * imag


np.random.seed(0)
np.set_printoptions(linewidth=200)

num_sites = 4

mat_a = get_random_op(num_sites)
mat_b = get_random_op(num_sites)
mat_c = commute_mats(mat_a, mat_b)

op_a = DenseMultiBodyOperators.from_matrix(mat_a, op_mats)
op_b = DenseMultiBodyOperators.from_matrix(mat_b, op_mats)
op_c = commute_dense_ops(op_a, op_b, structure_factors)

success = np.allclose(op_c.to_matrix(op_mats), mat_c)
print("SUCCESS" if success else "FAILURE")
exit()


####################################################################################################

op_mat = get_random_op(num_sites)
op_sum = DenseMultiBodyOperators.from_matrix(op_mat, op_mats)
op_poly = OperatorPolynomial.from_multi_body_ops(op_sum)


def factorization_rule(located_ops: MultiBodyOperator) -> OperatorPolynomial:
    output = OperatorPolynomial()
    factors = [MultiBodyOperator(located_op) for located_op in located_ops]
    product = ExpectationValueProduct(*factors)
    output.vec[product] = 1
    return output


op_poly = op_poly.factorize(factorization_rule)
print(op_poly**2)
exit()

state = np.random.random(2**num_sites) * np.exp(-1j * np.random.random(2**num_sites))
state = state / np.linalg.norm(state)