#!/usr/bin/env python3
import collections
import dataclasses
import functools
import itertools
from typing import Callable, Iterator, Sequence, TypeVar, Union

import numpy as np


def tensor_product(tensor_a: np.ndarray, tensor_b: np.ndarray) -> np.ndarray:
    return np.tensordot(tensor_a, tensor_b, axes=0)


####################################################################################################
# methods for computing structure factors: [A, B] = sum_C f_{ABC} C


def trace_inner_product(op_a: np.ndarray, op_b: np.ndarray) -> complex:
    return op_a.ravel().conj() @ op_b.ravel() / op_a.shape[0]


def multiply_mats(op_a: np.ndarray, op_b: np.ndarray) -> np.ndarray:
    return op_a @ op_b


def commute_mats(op_a: np.ndarray, op_b: np.ndarray) -> np.ndarray:
    return op_a @ op_b - op_b @ op_a


def get_structure_factors(
    op_mat_I: np.ndarray,
    *other_mats: np.ndarray,
    binary_op: Callable[[np.ndarray, np.ndarray], np.ndarray] = commute_mats,
) -> np.ndarray:
    dim = op_mat_I.shape[0]
    op_mats = [op_mat_I, *other_mats]
    assert np.array_equal(op_mat_I, np.eye(dim))
    assert len(op_mats) == dim**2

    structure_factors = np.empty((len(op_mats),) * 3, dtype=complex)
    for idx_a, mat_a in enumerate(other_mats, start=1):
        for idx_b, mat_b in enumerate(other_mats, start=1):
            mat_comm = binary_op(mat_a, mat_b)
            mat_comm_vec = [trace_inner_product(op_mat, mat_comm) for op_mat in op_mats]
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

    def __bool__(self) -> bool:
        return self.index != 0

    def __index__(self) -> int:
        return self.index

    def __lt__(self, other: "TypedInteger") -> bool:
        return self.index < other.index

    @classmethod
    def range(cls: type[Self], *args: int) -> Iterator[Self]:
        yield from (cls(index) for index in range(*args))


class SingleBodyOperator(TypedInteger):
    def __str__(self) -> str:
        return f"O_{self.index}"

    def as_matrix(self, op_mats: Sequence[np.ndarray]) -> np.ndarray:
        return op_mats[self.index]


class Site(TypedInteger):
    ...


####################################################################################################
# data structures for storing operators


@dataclasses.dataclass
class MultiBodyOperator:
    scalar: complex
    tensor: np.ndarray
    local_ops: tuple[SingleBodyOperator, ...]

    def __init__(
        self,
        tensor: np.ndarray,
        *local_ops: SingleBodyOperator,
        scalar: complex = 1,
        simplify: bool = True,
    ) -> None:
        if tensor.ndim and not tensor.ndim == len(local_ops):
            raise ValueError("tensor dimension does not match operator locality")
        self.scalar = scalar
        self.tensor = tensor
        self.local_ops = local_ops
        if simplify:
            self.simplify()

    def __str__(self) -> str:
        op_strs = " ".join(str(op) for op in self.local_ops)
        return f"D({op_strs})"

    def __mul__(self, scalar: complex) -> "MultiBodyOperator":
        new_scalar = self.scalar * scalar
        return MultiBodyOperator(self.tensor, *self.local_ops, scalar=new_scalar)

    def __rmul__(self, scalar: complex) -> "MultiBodyOperator":
        return self * scalar

    def __bool__(self) -> bool:
        return bool(self.scalar) and bool(self.tensor.any()) and bool(self.local_ops)

    def simplify(self) -> None:
        if any(not local_op for local_op in self.local_ops) and not self.is_identity_op:
            non_identity_data = [
                (idx, local_op) for idx, local_op in enumerate(self.local_ops) if local_op
            ]
            if non_identity_data:
                non_identity_indices, non_identity_ops = zip(*non_identity_data)
                self.tensor = np.einsum(self.tensor, range(self.tensor.ndim), non_identity_indices)
                self.local_ops = non_identity_ops
            else:
                self.local_ops = (SingleBodyOperator(0),) * self.num_sites
                self.scalar *= np.sum(self.tensor)
                self.tensor = np.array(1)

    @property
    def num_sites(self) -> int:
        if self.is_identity_op:
            return self.locality
        return self.tensor.shape[0]

    @property
    def locality(self) -> int:
        return len(self.local_ops)

    @property
    def is_identity_op(self) -> bool:
        return not self.tensor.ndim

    def in_canonical_form(self) -> "MultiBodyOperator":
        argsort = np.argsort([local_op.index for local_op in self.local_ops])
        tensor = np.transpose(self.tensor, argsort)
        local_ops = [self.local_ops[idx] for idx in argsort]
        return MultiBodyOperator(tensor, *local_ops, scalar=self.scalar)

    def as_tensor(self, op_mats: Sequence[np.ndarray]) -> np.ndarray:
        spin_dim = int(np.round(np.sqrt(len(op_mats))))
        op_mat_I = np.eye(spin_dim)
        assert np.array_equal(op_mats[0], op_mat_I)

        iden_op_tensor = _get_iden_op_tensor(spin_dim, self.num_sites - self.locality)
        local_op_tensors = [local_op.as_matrix(op_mats).ravel() for local_op in self.local_ops]
        base_tensor = self.scalar * functools.reduce(
            tensor_product, local_op_tensors + [iden_op_tensor]
        )

        if self.is_identity_op:
            return self.tensor * base_tensor

        op_tensors = (
            self.tensor[sites] * np.moveaxis(base_tensor, range(self.locality), sites)
            for sites in np.ndindex(self.tensor.shape)
            if self.tensor[sites]
        )
        return sum(op_tensors)


@functools.cache
def _get_iden_op_tensor(spin_dim: int, num_sites: int) -> np.ndarray:
    """Return a tensor product of flattened identity matrices."""
    iden_tensors = [np.eye(spin_dim).ravel()] * num_sites
    return functools.reduce(tensor_product, iden_tensors, np.array(1))


@dataclasses.dataclass
class MultiBodyOperators:
    ops: list[MultiBodyOperator]

    def __init__(
        self, *terms: Union[MultiBodyOperator, "MultiBodyOperators"], simplify: bool = True
    ) -> None:
        if terms:
            assert len(set(term.num_sites for term in terms)) == 1
        self.ops = [term for term in terms if isinstance(term, MultiBodyOperator)] + [
            op for term in terms if isinstance(term, MultiBodyOperators) for op in term.ops
        ]
        if simplify:
            self.simplify()

    def __mul__(self, scalar: complex) -> "MultiBodyOperators":
        new_ops = [scalar * op for op in self.ops]
        return MultiBodyOperators(*new_ops)

    def __rmul__(self, scalar: complex) -> "MultiBodyOperators":
        return self * scalar

    def __add__(
        self, other: Union[MultiBodyOperator, "MultiBodyOperators"]
    ) -> "MultiBodyOperators":
        assert self.num_sites == other.num_sites or self.num_sites == 0 or other.num_sites == 0
        return MultiBodyOperators(*self.ops, *MultiBodyOperators(other).ops)

    def __iadd__(
        self, other: Union[MultiBodyOperator, "MultiBodyOperators"]
    ) -> "MultiBodyOperators":
        self = self + other
        return self

    def __iter__(self) -> Iterator[MultiBodyOperator]:
        yield from self.ops

    @property
    def num_sites(self) -> int:
        if not self.ops:
            return 0
        return self.ops[0].num_sites

    def simplify(self) -> None:
        # remove trivial (zero) terms
        self.ops = [op for op in self.ops if op]
        # combine terms that are the same up to a permutation of local operators
        local_op_counts = [collections.Counter(op.local_ops) for op in self.ops]
        for jj in reversed(range(1, len(local_op_counts))):
            for ii in range(jj):
                if local_op_counts[ii] == local_op_counts[jj]:
                    op_ii = self.ops[ii].in_canonical_form()
                    op_jj = self.ops[jj].in_canonical_form()
                    local_ops = op_ii.local_ops
                    if np.array_equal(op_ii.tensor, op_jj.tensor):
                        tensor = op_ii.tensor
                        scalar = op_ii.scalar + op_jj.scalar
                        new_op = MultiBodyOperator(tensor, *local_ops, scalar=scalar)
                    else:
                        tensor = op_ii.tensor + op_jj.scalar / op_ii.scalar * op_jj.tensor
                        scalar = op_ii.scalar
                        new_op = MultiBodyOperator(tensor, *local_ops, scalar=scalar)
                    self.ops[ii] = new_op
                    del self.ops[jj]
                    break

    @classmethod
    def from_matrix(cls, matrix: np.ndarray, op_mats: Sequence[np.ndarray]) -> "MultiBodyOperators":
        spin_dim = int(np.round(np.sqrt(len(op_mats))))
        num_sites = int(np.round(np.log(matrix.size) / np.log(spin_dim))) // 2

        op_mat_I = np.eye(spin_dim)
        assert np.array_equal(op_mats[0], op_mat_I)

        output = MultiBodyOperators()
        identity_coefficient = trace_inner_product(np.eye(spin_dim**num_sites), matrix)
        if identity_coefficient:
            iden_ops = [SingleBodyOperator(0) for _ in range(num_sites)]
            output += MultiBodyOperator(np.array(1), *iden_ops, scalar=identity_coefficient)

        for non_iden_num in range(1, num_sites + 1):
            for non_iden_sites in itertools.combinations(range(num_sites), non_iden_num):
                tensor = np.zeros((num_sites,) * non_iden_num, dtype=complex)
                for local_ops in itertools.product(
                    SingleBodyOperator.range(1, len(op_mats)), repeat=non_iden_num
                ):
                    matrices = [op_mat_I] * num_sites
                    for idx, local_op in zip(non_iden_sites, local_ops):
                        matrices[idx] = local_op.as_matrix(op_mats)
                    pauli_op_matrix = functools.reduce(np.kron, matrices)
                    coefficient = trace_inner_product(pauli_op_matrix, matrix)
                    if coefficient:
                        tensor[non_iden_sites] = coefficient
                        output += MultiBodyOperator(tensor.copy(), *local_ops)
        return output

    def as_matrix(self, op_mats: Sequence[np.ndarray]) -> np.ndarray:
        spin_dim = int(np.round(np.sqrt(len(op_mats))))
        num_sites = self.num_sites
        final_shape = (spin_dim**num_sites,) * 2
        op_tensor = sum((op.as_tensor(op_mats) for op in self.ops))
        if not isinstance(op_tensor, np.ndarray):
            return np.zeros(final_shape)
        op_tensor.shape = (spin_dim,) * (2 * num_sites)
        op_tensor = np.moveaxis(
            op_tensor, range(1, 2 * num_sites, 2), range(num_sites, 2 * num_sites)
        )
        return op_tensor.reshape(final_shape)


####################################################################################################
# methods for commuting two 'MultiBodyOperators's.


def commute_ops(
    op_a: MultiBodyOperators | MultiBodyOperator,
    op_b: MultiBodyOperators | MultiBodyOperator,
    structure_factors: np.ndarray,
) -> MultiBodyOperators:
    op_a = MultiBodyOperators(op_a)
    op_b = MultiBodyOperators(op_b)
    output = MultiBodyOperators()
    for term_a, term_b in itertools.product(op_a.ops, op_b.ops):
        output += _commute_op_terms(term_a, term_b, structure_factors)
    output.simplify()
    return output


def _commute_op_terms(
    op_a: MultiBodyOperator,
    op_b: MultiBodyOperator,
    structure_factors: np.ndarray,
) -> MultiBodyOperators:
    output = MultiBodyOperators()
    op_dim = structure_factors.shape[-1]

    for num_overlaps in range(1, min(op_a.locality, op_b.locality) + 1):
        overlap_index_set_a = itertools.combinations(range(op_a.locality), num_overlaps)
        overlap_index_set_b = (
            permuted_indices
            for indices in itertools.combinations(range(op_b.locality), num_overlaps)
            for permuted_indices in itertools.permutations(indices)
        )

        for overlap_indices_a, overlap_indices_b in itertools.product(
            overlap_index_set_a, overlap_index_set_b
        ):
            # construct the tensor of coefficients for terms with these overlaps
            indices_a = list(range(op_a.locality))
            indices_b = list(range(op_a.locality, op_a.locality + op_b.locality))
            for idx_a, idx_b in zip(overlap_indices_a, overlap_indices_b):
                indices_b[idx_b] = idx_a
            final_indices = indices_a + [idx for idx in indices_b if idx >= op_a.locality]
            tensor = np.einsum(op_a.tensor, indices_a, op_b.tensor, indices_b, final_indices)

            # construct a tentative list of local operators for terms with these overlaps
            local_ops_a = list(op_a.local_ops)
            local_ops_b = [
                op for idx, op in enumerate(op_b.local_ops) if idx not in overlap_indices_b
            ]
            local_ops = local_ops_a + local_ops_b

            # construct lists of the local operators that will overlap
            overlap_ops_a = [op_a.local_ops[idx] for idx in overlap_indices_a]
            overlap_ops_b = [op_b.local_ops[idx] for idx in overlap_indices_b]

            # loop over terms in the commutator of 'operlap_ops_a' and 'overlap_ops_b'
            for overlap_ops in itertools.product(
                SingleBodyOperator.range(op_dim), repeat=num_overlaps
            ):
                # get the structure factors for this term
                factors = [
                    structure_factors[aa, bb, cc]
                    for aa, bb, cc in zip(overlap_ops_a, overlap_ops_b, overlap_ops)
                ]
                if not all(factors):
                    continue

                # update the list of all local operators in this term
                for idx, overlap_op in enumerate(overlap_ops):
                    local_ops[overlap_indices_a[idx]] = overlap_op

                # add this term to the output
                scalar = np.prod(factors) * op_a.scalar * op_b.scalar
                output += MultiBodyOperator(tensor, *local_ops, scalar=scalar)

    output.simplify()
    return output


####################################################################################################
# data structures to represent polynomials of "located" operators


@dataclasses.dataclass
class LocatedOperator:
    """A product of single-body operators located at specific sites."""

    _data: frozenset[tuple[SingleBodyOperator, Site]]

    def __init__(self, *ops_and_sites: tuple[SingleBodyOperator, Site]) -> None:
        self._data = frozenset(ops_and_sites)

    def __str__(self) -> str:
        return " ".join(f"{op}({site})" for op, site in sorted(self._data))

    def __hash__(self) -> int:
        return hash(str(self))

    def __iter__(self) -> Iterator[tuple[SingleBodyOperator, Site]]:
        yield from self._data

    def __mul__(self, other: "LocatedOperator") -> "LocatedOperatorProduct":
        return LocatedOperatorProduct((self, 1)) * LocatedOperatorProduct((other, 1))


class LocatedOperatorProduct:
    """A product of expectation values of 'LocatedOperator's."""

    _op_to_exp: collections.defaultdict[LocatedOperator, int]

    def __init__(self, *ops_and_exponents: tuple[LocatedOperator, int]) -> None:
        self._op_to_exp = collections.defaultdict(int)
        self._op_to_exp.update(dict(ops_and_exponents))

    def __str__(self) -> str:
        return " ".join(f"<{op}>" if exp == 1 else f"<{op}>**{exp}" for op, exp in self)

    def __hash__(self) -> int:
        return hash(frozenset(self._op_to_exp.items()))

    def __iter__(self) -> Iterator[tuple[LocatedOperator, int]]:
        yield from self._op_to_exp.items()

    def __mul__(
        self, other: Union[LocatedOperator, "LocatedOperatorProduct"]
    ) -> "LocatedOperatorProduct":
        output = LocatedOperatorProduct()
        output._op_to_exp = self._op_to_exp.copy()
        if isinstance(other, LocatedOperator):
            output._op_to_exp[other] += 1
        else:
            for op, exp in other:
                output._op_to_exp[op] += exp
        return output

    def __rmul__(
        self, other: Union[LocatedOperator, "LocatedOperatorProduct"]
    ) -> "LocatedOperatorProduct":
        return self * other


@dataclasses.dataclass
class OperatorPolynomial:
    """A polynomial of expectation values of 'LocatedOperator's."""

    vec: dict[LocatedOperatorProduct, complex]

    def __init__(self, *terms: MultiBodyOperators | MultiBodyOperator) -> None:
        self.vec = collections.defaultdict(complex)
        for op_sum in terms:
            if isinstance(op_sum, MultiBodyOperator):
                op_sum = MultiBodyOperators(op_sum)
            op_sum.simplify()

            for op in op_sum.ops:
                if op.is_identity_op:
                    located_op = LocatedOperator(*zip(op.local_ops, Site.range(op.num_sites)))
                    term = LocatedOperatorProduct((located_op, 1))
                    self.vec[term] += op.scalar * complex(op.tensor)
                    continue

                for addressed_sites in itertools.combinations(
                    Site.range(op.num_sites), op.locality
                ):
                    for op_sites in itertools.permutations(addressed_sites):
                        if op.tensor[op_sites]:
                            located_op = LocatedOperator(*zip(op.local_ops, op_sites))
                            term = LocatedOperatorProduct((located_op, 1))
                            self.vec[term] += op.scalar * op.tensor[op_sites]

    def __str__(self) -> str:
        return "\n".join(f"{scalar} {op}" for op, scalar in self)

    def __iter__(self) -> Iterator[tuple[LocatedOperatorProduct, complex]]:
        yield from self.vec.items()

    def __add__(self, other: "OperatorPolynomial") -> "OperatorPolynomial":
        output = OperatorPolynomial()
        output.vec = self.vec.copy()
        for term, scalar in other:
            output.vec[term] += scalar
        return output

    def __mul__(self, other: Union[complex, "OperatorPolynomial"]) -> "OperatorPolynomial":
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
        assert exponent > 0
        output = OperatorPolynomial()
        for _ in range(exponent):
            output = output * self
        return output

    def factorize(
        self, factorize: Callable[[LocatedOperator], "OperatorPolynomial"]
    ) -> "OperatorPolynomial":
        output = OperatorPolynomial()
        for term, scalar in self:
            factorized_factors = [factorize(factor) ** exponent for factor, exponent in term]
            product_of_factorized_factors = functools.reduce(
                OperatorPolynomial.__mul__, factorized_factors
            )
            output += scalar * product_of_factorized_factors
        return output


####################################################################################################

sites = 3

op_I = SingleBodyOperator(0)
op_Z = SingleBodyOperator(1)
op_X = SingleBodyOperator(2)
op_Y = SingleBodyOperator(3)
ops = [op_I, op_Z, op_X, op_Y]

op_mat_I = np.eye(2)
op_mat_Z = np.array([[1, 0], [0, -1]])
op_mat_X = np.array([[0, 1], [1, 0]])
op_mat_Y = -1j * op_mat_Z @ op_mat_X
op_mats = [op_mat_I, op_mat_Z, op_mat_X, op_mat_Y]

structure_factors = get_structure_factors(*op_mats)

tensor_a = np.ones(sites)
tensor_b = np.ones((sites, sites))
tensor_c = np.ones((sites, sites, sites))

for ii in range(sites):
    tensor_b[ii, ii] = 0
    tensor_c[:, ii, ii] = tensor_c[ii, :, ii] = tensor_c[ii, ii, :] = 0

op_a = MultiBodyOperator(tensor_a, op_Z)
op_b = MultiBodyOperator(tensor_b, op_Z, op_X)
op_c = MultiBodyOperator(tensor_c, op_Y, op_Z, op_X)

op_joined = commute_ops(op_a, op_c, structure_factors)

for op in op_joined:
    print(op.scalar, op)
    print()

####################################################################################################

np.random.seed(0)
np.set_printoptions(linewidth=200)

num_sites = 3
hamiltonian = np.random.random((2**num_sites,) * 2) + 1j * np.random.random((2**num_sites,) * 2)
hamiltonian = hamiltonian + hamiltonian.conj().T

op_sum = MultiBodyOperators.from_matrix(hamiltonian, op_mats)
op_mat = op_sum.as_matrix(op_mats)
success = np.allclose(op_mat, hamiltonian)
print("SUCCESS" if success else "FAILURE")
print()

####################################################################################################

op_poly = OperatorPolynomial(op_sum)
print(op_poly)

exit()

state = np.random.random(2**num_sites) * np.exp(-1j * np.random.random(2**num_sites))
state = state / np.linalg.norm(state)
