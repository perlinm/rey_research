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

    def __init__(self, index: int) -> None:
        self.index = int(index)

    def __str__(self) -> str:
        return str(self.index)

    def __hash__(self) -> int:
        return hash(str(self))

    def __index__(self) -> int:
        return self.index

    def __lt__(self, other: type[Self]) -> bool:
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

    def __lt__(self, other: "SingleBodyOperator") -> bool:
        return self.site.index < other.site.index

    def is_identity_op(self) -> bool:
        return self.op.is_identity_op()


@dataclasses.dataclass
class MultiBodyOperator:
    """A product of (non-identity) single-body operators located at specific sites."""

    ops: frozenset[SingleBodyOperator]

    def __init__(self, *located_ops: SingleBodyOperator) -> None:
        self.ops = frozenset(op for op in located_ops if not op.is_identity_op())
        assert len(located_ops) == len(set(op.site for op in located_ops))

    def __str__(self) -> str:
        return " ".join(str(op) for op in self)

    def __hash__(self) -> int:
        return hash(str(self))

    def __iter__(self) -> Iterator[SingleBodyOperator]:
        yield from sorted(self.ops)

    def __bool__(self) -> bool:
        return bool(self.ops)

    @property
    def locality(self) -> int:
        return len(self.ops)

    def is_identity_op(self) -> bool:
        return len(self.ops) == 0


@dataclasses.dataclass
class DenseMultiBodyOperator:
    dist_ops: tuple[AbstractSingleBodyOperator, ...]
    tensor: np.ndarray
    scalar: complex
    fixed_op: MultiBodyOperator

    def __init__(
        self,
        *dist_ops: AbstractSingleBodyOperator,
        tensor: np.ndarray = np.array(1),
        scalar: complex = 1,
        fixed_op: MultiBodyOperator = MultiBodyOperator(),
        num_sites: Optional[int] = None,
        simplify: bool = True,
    ) -> None:
        if tensor.ndim != len(dist_ops):
            raise ValueError("tensor dimension does not match operator locality")
        self.scalar = scalar
        self.tensor = tensor
        self.dist_ops = dist_ops
        self.fixed_op = fixed_op

        if num_sites is not None:
            self._num_sites = num_sites
        elif tensor.ndim:
            self._num_sites = tensor.shape[0]
        else:
            raise ValueError("not enough data provided to determine operator locality")
        if tensor.ndim:
            assert all(dim == self._num_sites for dim in tensor.shape)

        self._locality = self.fixed_op.locality + len(
            [op for op in dist_ops if not op.is_identity_op()]
        )

        if simplify:
            self.simplify()

    def __str__(self) -> str:
        op_strs = []
        if self.fixed_op:
            op_strs.append(f"F[{self.fixed_op}]")
        if self.dist_ops:
            dist_op_str = " ".join(str(op) for op in self.dist_ops)
            op_strs.append(f"D({dist_op_str})")
        return "⋅".join(op_strs)

    def __mul__(self, scalar: complex) -> "DenseMultiBodyOperator":
        new_scalar = self.scalar * scalar
        return DenseMultiBodyOperator(*self.dist_ops, tensor=self.tensor, scalar=new_scalar)

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

    @property
    def fixed_ops(self) -> tuple[AbstractSingleBodyOperator, ...]:
        return tuple(located_op.op for located_op in self.fixed_op)

    @property
    def fixed_sites(self) -> tuple[LatticeSite, ...]:
        return tuple(located_op.site for located_op in self.fixed_op)

    @property
    def local_ops(self) -> tuple[AbstractSingleBodyOperator, ...]:
        return self.fixed_ops + self.dist_ops

    def simplify(self) -> None:
        """Simplify 'self' by removing any identity operators from 'self.dist_ops'."""
        # identity all nontrivial (non-identity) local operators
        non_identity_data = [
            (idx, local_op)
            for idx, local_op in enumerate(self.dist_ops)
            if not local_op.is_identity_op()
        ]
        if len(non_identity_data) != self.tensor.ndim:
            # trace over the indices in the tensor associated with identity operators
            if non_identity_data:
                non_identity_indices, non_identity_ops = zip(*non_identity_data)
            else:
                non_identity_indices, non_identity_ops = (), ()
            self.tensor = np.einsum(self.tensor, range(self.tensor.ndim), non_identity_indices)
            self.dist_ops = non_identity_ops

    def in_canonical_form(self) -> "DenseMultiBodyOperator":
        """Rearrange data to sort 'self.dist_ops' by increasing index."""
        argsort = np.argsort([local_op.index for local_op in self.dist_ops])
        tensor = np.transpose(self.tensor, argsort)
        dist_ops = [self.dist_ops[idx] for idx in argsort]
        return DenseMultiBodyOperator(
            *dist_ops,
            tensor=tensor,
            scalar=self.scalar,
            fixed_op=self.fixed_op,
            num_sites=self.num_sites,
        )

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
        iden_op_tensor = _get_iden_op_tensor(spin_dim, self.num_sites - self.locality)

        # vectorize all nontrivial operators in 'self', and identify fixed/nonfixed sites
        self.simplify()
        local_op_vecs = tuple(op_mats[local_op.index].ravel() for local_op in self.dist_ops)
        if self.fixed_op:
            fixed_op_vecs, fixed_op_sites = zip(
                *[(op_mats[op.op.index].ravel(), op.site) for op in self.fixed_op]
            )
        else:
            fixed_op_vecs, fixed_op_sites = (), ()
        non_fixed_op_sites = [
            site for site in LatticeSite.range(self.num_sites) if site not in fixed_op_sites
        ]

        # take a tensor product of vectorized operators for all sites
        base_tensor = self.scalar * functools.reduce(
            tensor_product, fixed_op_vecs + local_op_vecs + (iden_op_tensor,)
        )

        # sum over all choices of sites to address nontrivially
        op_tensors = (
            self.tensor[dist_op_sites]
            * np.moveaxis(base_tensor, range(self.locality), fixed_op_sites + dist_op_sites)
            for non_fixed_sites in itertools.combinations(non_fixed_op_sites, self.tensor.ndim)
            for dist_op_sites in itertools.permutations(non_fixed_sites)
            if self.tensor[dist_op_sites]
        )
        return sum(op_tensors)

    def to_matrix(self, op_mats: Sequence[np.ndarray]) -> np.ndarray:
        """Return a matrix representation of 'self'."""
        spin_dim = int(np.round(np.sqrt(len(op_mats))))
        num_sites = self.num_sites
        output_matrix_shape = (spin_dim**num_sites,) * 2
        if not self:
            return np.zeros(output_matrix_shape)
        split_tensor_shape = (spin_dim,) * (2 * num_sites)
        return np.moveaxis(
            self.to_tensor(op_mats).reshape(split_tensor_shape),
            range(1, 2 * num_sites, 2),
            range(num_sites, 2 * num_sites),
        ).reshape(output_matrix_shape)


@functools.cache
def _get_iden_op_tensor(spin_dim: int, num_sites: int) -> np.ndarray:
    """Return a tensor product of flattened identity matrices."""
    iden_tensors = [np.eye(spin_dim).ravel()] * num_sites
    return functools.reduce(tensor_product, iden_tensors, np.array(1))


@dataclasses.dataclass
class DenseMultiBodyOperators:
    terms: list[DenseMultiBodyOperator]

    def __init__(
        self,
        *terms: Union[DenseMultiBodyOperator, "DenseMultiBodyOperators"],
        simplify: bool = True,
    ) -> None:
        if terms:
            assert len(set(term.num_sites for term in terms)) == 1
        self.terms = [term for term in terms if isinstance(term, DenseMultiBodyOperator)]
        for term in terms:
            if isinstance(term, DenseMultiBodyOperators):
                self.terms.extend(term.terms)
        if simplify:
            self.simplify()

    def __mul__(self, scalar: complex) -> "DenseMultiBodyOperators":
        new_terms = [scalar * term for term in self.terms]
        return DenseMultiBodyOperators(*new_terms)

    def __rmul__(self, scalar: complex) -> "DenseMultiBodyOperators":
        return self * scalar

    def __add__(
        self, other: Union[DenseMultiBodyOperator, "DenseMultiBodyOperators"]
    ) -> "DenseMultiBodyOperators":
        assert self.num_sites == other.num_sites or self.num_sites == 0 or other.num_sites == 0
        if isinstance(other, DenseMultiBodyOperator):
            other = DenseMultiBodyOperators(other, simplify=False)
        return DenseMultiBodyOperators(*self.terms, *other.terms, simplify=False)

    def __iadd__(
        self, other: Union[DenseMultiBodyOperator, "DenseMultiBodyOperators"]
    ) -> "DenseMultiBodyOperators":
        self = self + other
        return self

    @property
    def num_sites(self) -> int:
        if not self.terms:
            return 0
        return self.terms[0].num_sites

    def _pin_local_terms(self) -> None:
        """
        If any term has a tensor with only one nonzero element, then "pin" the operators in that
        term, fixing them to lattice sites.
        """
        for ii in range(len(self.terms) - 1, -1, -1):
            term = self.terms[ii]
            if len(nonzero_site_indices := np.argwhere(term.tensor)) == 1:
                site_indices = tuple(nonzero_site_indices[0])
                if len(set(site_indices)) != len(site_indices) or any(
                    site.index in site_indices for site in term.fixed_sites
                ):
                    del self.terms[ii]
                    continue

                scalar = term.scalar * term.tensor[site_indices]
                fixed_op_list = [
                    SingleBodyOperator(local_op, LatticeSite(site_index))
                    for local_op, site_index in zip(term.dist_ops, site_indices)
                ]
                fixed_op = MultiBodyOperator(*fixed_op_list, *term.fixed_op)
                self.terms[ii] = DenseMultiBodyOperator(
                    scalar=scalar, fixed_op=fixed_op, num_sites=self.num_sites
                )

    def _combine_similar_terms(self) -> None:
        """Combine terms in 'self' that are the same up to a permutatino of local operators."""
        local_op_counts = [collections.Counter(op.dist_ops) for op in self.terms]
        for jj in range(len(local_op_counts) - 1, 0, -1):
            for ii in range(jj):
                # if terms ii and jj have the same operator content, merge term jj into ii
                if (
                    self.terms[ii].fixed_op == self.terms[jj].fixed_op
                    and local_op_counts[ii] == local_op_counts[jj]
                ):
                    term_ii = self.terms[ii].in_canonical_form()
                    term_jj = self.terms[jj].in_canonical_form()
                    dist_ops = term_ii.dist_ops
                    if term_ii.tensor is term_jj.tensor:
                        tensor = term_ii.tensor
                        scalar = term_ii.scalar + term_jj.scalar
                    else:
                        tensor = term_ii.tensor + term_jj.scalar / term_ii.scalar * term_jj.tensor
                        scalar = term_ii.scalar
                    new_op = DenseMultiBodyOperator(
                        *dist_ops,
                        tensor=tensor,
                        scalar=scalar,
                        num_sites=self.num_sites,
                        fixed_op=self.terms[ii].fixed_op,
                    )
                    self.terms[ii] = new_op
                    del self.terms[jj]
                    break

    def simplify(self) -> None:
        """
        Simplify 'self' by removing trivial (zero) terms, and combining terms that have the same
        operator content.
        """
        # remove trivial (zero) terms
        self.terms = [term for term in self.terms if bool(term)]

        self._combine_similar_terms()
        self._pin_local_terms()
        self._combine_similar_terms()

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
            output += DenseMultiBodyOperator(scalar=identity_coefficient, num_sites=num_sites)

        # loop over all numbers of sites that may be addressed nontrivially
        for locality in range(1, num_sites + 1):

            # loop over all choices of specific sites that are addressed nontrivially
            for non_iden_sites in itertools.combinations(range(num_sites), locality):

                # initialize a coefficient tensor to populate with nonzero values
                tensor = np.zeros((num_sites,) * locality, dtype=complex)

                # loop over all choices of nontrivial operators at the chosen sites
                for dist_ops in itertools.product(
                    AbstractSingleBodyOperator.range(1, len(op_mats)), repeat=locality
                ):
                    # compute the coefficient for this choice of local operators
                    matrices = [op_mat_I] * num_sites
                    for idx, local_op in zip(non_iden_sites, dist_ops):
                        matrices[idx] = op_mats[local_op.index]
                    term_matrix = functools.reduce(np.kron, matrices)
                    coefficient = trace_inner_product(term_matrix, matrix)

                    # add this term to the output
                    if coefficient:
                        tensor[non_iden_sites] = 1
                        output += DenseMultiBodyOperator(
                            *dist_ops,
                            tensor=tensor.copy(),
                            scalar=coefficient,
                            num_sites=num_sites,
                        )

        output.simplify()
        return output

    def to_matrix(self, op_mats: Sequence[np.ndarray]) -> np.ndarray:
        """Return the matrix representation of 'self'."""
        self.simplify()
        matrix = sum((term.to_matrix(op_mats) for term in self.terms))
        if not isinstance(matrix, np.ndarray):
            spin_dim = int(np.round(np.sqrt(len(op_mats))))
            num_sites = self.num_sites
            output_matrix_shape = (spin_dim**num_sites,) * 2
            return np.zeros(output_matrix_shape)
        return matrix


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
    for term_a, term_b in itertools.product(op_a.terms, op_b.terms):
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
        overlap_ops_a: Sequence[AbstractSingleBodyOperator],
        overlap_ops_b: Sequence[AbstractSingleBodyOperator],
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
    output = DenseMultiBodyOperators()

    if op_a.is_identity_op() or op_b.is_identity_op():
        return output

    # loop over all numbers of overlapping local operators between 'op_a' and 'op_b'
    min_overlaps = max(1, op_a.locality + op_b.locality - op_a.num_sites)
    max_overlaps = min(op_a.locality, op_b.locality)
    for num_overlaps in range(min_overlaps, max_overlaps + 1):

        # loop over all choices of local operators in 'op_a' to overlap with 'op_b'
        index_choices_a = itertools.combinations(range(op_a.locality), num_overlaps)
        for overlap_indices_a in index_choices_a:

            # identify overlap data for 'op_a'
            (
                overlap_ops_a,
                dist_non_overlap_ops_a,
                fixed_non_overlap_ops_a,
                fixed_overlap_sites_a,
            ) = _get_overlap_data(overlap_indices_a, op_a.dist_ops, op_a.fixed_op)

            # loop over choices of local operators in 'op_b' to overlap with 'op_a'
            index_choices_b = _get_overlap_index_choices(
                overlap_indices_a, op_a.fixed_sites, op_b.fixed_sites, op_b.locality
            )
            for overlap_indices_b in index_choices_b:

                # construct the tensor of coefficients for this choice of overlaps
                overlap_tensor = _get_overlap_tensor(
                    op_a.tensor,
                    op_b.tensor,
                    overlap_indices_a,
                    overlap_indices_b,
                    op_a.fixed_sites,
                    op_b.fixed_sites,
                )
                if not overlap_tensor.any():
                    continue

                # identify overlap data for 'op_b'
                (
                    overlap_ops_b,
                    dist_non_overlap_ops_b,
                    fixed_non_overlap_ops_b,
                    fixed_overlap_sites_b,
                ) = _get_overlap_data(overlap_indices_b, op_b.dist_ops, op_b.fixed_op)

                # collect fixed operator/site data
                fixed_non_overlap_ops = fixed_non_overlap_ops_a + fixed_non_overlap_ops_b
                fixed_overlap_sites = tuple(
                    op_a.fixed_sites[idx_a]
                    if idx_a < op_a.fixed_op.locality
                    else op_b.fixed_sites[idx_b]
                    if idx_b < op_b.fixed_op.locality
                    else None
                    for idx_a, idx_b in zip(overlap_indices_a, overlap_indices_b)
                )

                # add all nonzero terms in the commutator with these overlaps
                commutator_factors = commutator_factor_func(overlap_ops_a, overlap_ops_b)
                for overlap_ops_by_index in np.argwhere(commutator_factors):
                    commutator_factor = commutator_factors[tuple(overlap_ops_by_index)]

                    # identify the operator content of this term in the commutator
                    overlap_dist_ops, overlap_fixed_op = _get_commutator_term_ops(
                        overlap_ops_by_index, fixed_overlap_sites, fixed_non_overlap_ops
                    )

                    output += DenseMultiBodyOperator(
                        *overlap_dist_ops,
                        *dist_non_overlap_ops_a,
                        *dist_non_overlap_ops_b,
                        tensor=overlap_tensor,
                        scalar=op_a.scalar * op_b.scalar * commutator_factor,
                        fixed_op=overlap_fixed_op,
                        num_sites=op_a.num_sites,
                    )

    return output


def _get_overlap_index_choices(
    other_overlap_indices: tuple[int, ...],
    other_fixed_sites: tuple[LatticeSite, ...],
    fixed_sites: tuple[LatticeSite, ...],
    locality: int,
) -> Iterator[tuple[int, ...]]:
    """Iterate over choices of local operators to overlap with another operator."""
    num_overlaps = len(other_overlap_indices)
    fixed_op_locality = len(fixed_sites)
    other_fixed_op_locality = len(other_fixed_sites)

    check_for_invalid_fixed_op_overlaps = fixed_op_locality and other_fixed_op_locality

    # identify fixed ops that *must* overlap
    overlap_indices = np.zeros(num_overlaps, dtype=int) - 1
    for overlap_idx, op_idx in enumerate(other_overlap_indices):
        if op_idx >= other_fixed_op_locality:
            break
        fixed_site = other_fixed_sites[op_idx]
        if fixed_site in fixed_sites:
            overlap_indices[overlap_idx] = fixed_sites.index(fixed_site)

    # identify available indices and where they must be located
    available_indices = [idx for idx in range(locality) if idx not in overlap_indices]
    vacant_index_locs = np.argwhere(overlap_indices == -1).flatten()

    if check_for_invalid_fixed_op_overlaps:
        # if any non-overlapping fixed ops share sites, then there are no valid choices
        non_overlapping_fixed_sites = set(
            fixed_sites[idx] for idx in available_indices if idx < len(fixed_sites)
        )
        other_non_overlapping_fixed_sites = set(
            site for idx, site in enumerate(other_fixed_sites) if idx not in other_overlap_indices
        )
        if non_overlapping_fixed_sites & other_non_overlapping_fixed_sites:
            return

    # iterate over choices of indices to overlap!
    for indices in itertools.combinations(available_indices, len(vacant_index_locs)):
        for permuted_indices in itertools.permutations(indices):
            overlap_indices[vacant_index_locs] = permuted_indices
            # skip choices that overlap fixed operators at different sites
            if check_for_invalid_fixed_op_overlaps and (
                any(
                    idx < fixed_op_locality
                    and other_idx < other_fixed_op_locality
                    and fixed_sites[idx] != other_fixed_sites[other_idx]
                    for idx, other_idx in zip(overlap_indices, other_overlap_indices)
                )
            ):
                continue
            yield tuple(overlap_indices)


def _get_overlap_data(
    overlap_indices: tuple[int, ...],
    dist_ops: tuple[AbstractSingleBodyOperator, ...],
    fixed_op: MultiBodyOperator,
) -> tuple[
    tuple[AbstractSingleBodyOperator, ...],
    tuple[AbstractSingleBodyOperator, ...],
    tuple[SingleBodyOperator, ...],
    tuple[LatticeSite, ...],
]:
    """
    Identify:
    - The local operators that *will* overlap in these terms.
    - The fixed operators that *will not* overlap in these terms.
    - The distributed operators that *will not* overlap in these terms.
    - The lattice sites of fixed operators that *will* overlap.
    """
    if fixed_op:
        fixed_ops, fixed_sites = zip(*[(located_op.op, located_op.site) for located_op in fixed_op])
    else:
        fixed_ops, fixed_sites = (), ()
    local_ops = fixed_ops + dist_ops

    # identify the local operators in 'dense_op' that will/won't overlap in these terms
    overlap_ops = tuple(local_ops[idx] for idx in overlap_indices)
    dist_non_overlap_ops = tuple(
        op for idx, op in enumerate(dist_ops) if idx + len(fixed_ops) not in overlap_indices
    )
    fixed_non_overlap_ops = tuple(
        op for idx, op in enumerate(fixed_op) if idx not in overlap_indices
    )

    # identify sites of fixed operators that overlap
    fixed_overlap_sites = tuple(fixed_sites[idx] for idx in overlap_indices if idx < len(fixed_ops))

    return overlap_ops, dist_non_overlap_ops, fixed_non_overlap_ops, fixed_overlap_sites


def _get_overlap_tensor(
    tensor_a: np.ndarray,
    tensor_b: np.ndarray,
    overlap_indices_a: tuple[int, ...],
    overlap_indices_b: tuple[int, ...],
    fixed_sites_a: tuple[LatticeSite, ...],
    fixed_sites_b: tuple[LatticeSite, ...],
) -> np.ndarray:
    """
    Construct the tensor of coefficients for a given choice of local operators to overlap between
    two 'DenseMultiBodyOperator's.
    """
    num_sites = tensor_a.shape[0] if tensor_a.ndim else tensor_b.shape[0] if tensor_b.ndim else 0
    fixed_op_locality_a = len(fixed_sites_a)
    fixed_op_locality_b = len(fixed_sites_b)

    # identify indices for einsum expression
    tensor_indices_a = list(range(tensor_a.ndim))
    tensor_indices_b = [idx + tensor_a.ndim for idx in range(tensor_b.ndim)]
    for idx_a, idx_b in zip(overlap_indices_a, overlap_indices_b):
        tensor_index_a = idx_a - fixed_op_locality_a
        tensor_index_b = idx_b - fixed_op_locality_b
        if tensor_index_a >= 0 and tensor_index_b >= 0:
            tensor_indices_b[tensor_index_b] = tensor_index_a

    # identify how to slice tensors to fix indices that overlap with fixed operators
    tensor_slices_a = [slice(dim) for dim in tensor_a.shape]
    tensor_slices_b = [slice(dim) for dim in tensor_b.shape]
    for idx_a, idx_b in zip(overlap_indices_a, overlap_indices_b):
        tensor_index_a = idx_a - fixed_op_locality_a
        tensor_index_b = idx_b - fixed_op_locality_b
        if tensor_index_a >= 0 and idx_b < fixed_op_locality_b:
            val = fixed_sites_b[idx_b].index
            tensor_slices_a[tensor_index_a] = slice(val, val + 1)
        if tensor_index_b >= 0 and idx_a < fixed_op_locality_a:
            val = fixed_sites_a[idx_a].index
            tensor_slices_b[tensor_index_b] = slice(val, val + 1)

    # collect indices of the combined tensor
    final_indices_a = [
        idx
        for idx, idx_slice in zip(tensor_indices_a, tensor_slices_a)
        if idx_slice == slice(num_sites)
    ]
    final_indices_b = [
        idx
        for idx, idx_slice in zip(tensor_indices_b, tensor_slices_b)
        if idx not in tensor_indices_a and idx_slice == slice(num_sites)
    ]
    final_indices = (
        [idx for idx in final_indices_a if idx + fixed_op_locality_a in overlap_indices_a]
        + [idx for idx in final_indices_a if idx + fixed_op_locality_a not in overlap_indices_a]
        + final_indices_b
    )

    # combine tensors with an einsum expression
    overlap_tensor = np.einsum(
        tensor_a[tuple(tensor_slices_a)],
        tensor_indices_a,
        tensor_b[tuple(tensor_slices_b)],
        tensor_indices_b,
        final_indices,
    )
    if overlap_tensor.ndim == 0:
        return overlap_tensor
    if overlap_tensor.shape == (1,):
        return np.array(overlap_tensor[0])

    # remove entries with nonzero coefficients for two ops to address the same site
    num_sites = overlap_tensor.shape[0]
    for ii, jj in itertools.combinations(range(overlap_tensor.ndim), 2):
        indices: list[slice | int] = [slice(num_sites) for _ in overlap_tensor.shape]
        for site_index in range(num_sites):
            indices[ii] = indices[jj] = site_index
            overlap_tensor[tuple(indices)] = 0
    for ii in range(overlap_tensor.ndim):
        indices = [slice(num_sites) for _ in overlap_tensor.shape]
        for site in fixed_sites_a + fixed_sites_b:
            indices[ii] = site.index
            overlap_tensor[tuple(indices)] = 0

    return overlap_tensor


def _get_commutator_term_ops(
    overlap_ops_by_index: tuple[int, ...],
    fixed_overlap_sites: tuple[Optional[LatticeSite], ...],
    fixed_non_overlap_ops: tuple[SingleBodyOperator, ...],
) -> tuple[tuple[AbstractSingleBodyOperator, ...], MultiBodyOperator]:
    """Identify the operator content of a particular term in a commutator."""
    overlap_dist_ops = tuple(
        AbstractSingleBodyOperator(cc)
        for cc, site in zip(overlap_ops_by_index, fixed_overlap_sites)
        if site is None
    )
    overlap_fixed_ops = tuple(
        SingleBodyOperator(AbstractSingleBodyOperator(cc), site)
        for cc, site in zip(overlap_ops_by_index, fixed_overlap_sites)
        if site is not None
    )
    overlap_fixed_op = MultiBodyOperator(*overlap_fixed_ops, *fixed_non_overlap_ops)
    return overlap_dist_ops, overlap_fixed_op


####################################################################################################
# data structures to represent expectation values and polynomials thereof


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
    def from_multi_body_ops(
        self,
        *terms: DenseMultiBodyOperators | DenseMultiBodyOperator,
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
                    for site in LatticeSite.range(dense_op.num_sites)
                    if site not in dense_op.fixed_sites
                ]
                for addressed_sites in itertools.combinations(
                    available_sites, len(dense_op.dist_ops)
                ):
                    # loop over all assignments of specific operators to specific sites
                    for op_sites in itertools.permutations(addressed_sites):
                        if dense_op.tensor[op_sites]:
                            dist_ops = [
                                SingleBodyOperator(dist_op, site)
                                for dist_op, site in zip(dense_op.dist_ops, op_sites)
                            ]
                            multi_body_op = MultiBodyOperator(*dist_ops, *fixed_ops)
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
