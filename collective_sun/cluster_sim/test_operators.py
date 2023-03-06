#!/usr/bin/env python3
import functools
import itertools
from typing import Iterator, Optional, Sequence

import numpy as np

import operators


np.random.seed(0)
np.set_printoptions(linewidth=200, precision=3)


op_I = operators.AbstractSingleBodyOperator(0)
op_Z = operators.AbstractSingleBodyOperator(1)
op_X = operators.AbstractSingleBodyOperator(2)
op_Y = operators.AbstractSingleBodyOperator(3)
ops = [op_I, op_Z, op_X, op_Y]

op_mat_I = np.eye(2, dtype=complex)
op_mat_Z = np.array([[1, 0], [0, -1]], dtype=complex)
op_mat_X = np.array([[0, 1], [1, 0]], dtype=complex)
op_mat_Y = -1j * op_mat_Z @ op_mat_X
op_mats = [op_mat_I, op_mat_Z, op_mat_X, op_mat_Y]

structure_factors = operators.get_structure_factors(*op_mats)


def get_random_op(num_sites: int) -> np.ndarray:
    real = np.random.random((2**num_sites,) * 2)
    imag = np.random.random((2**num_sites,) * 2)
    return real + 1j * imag


def select_terms(
    matrix: np.ndarray, terms: Sequence[int], num_sites: Optional[int] = None
) -> np.ndarray:
    if num_sites is None:
        num_sites = int(np.round(np.log2(matrix.shape[0])))
    max_term = max(terms)
    new_matrix = np.zeros_like(matrix)
    for term, mats in enumerate(itertools.product(op_mats, repeat=num_sites)):
        if term in terms:
            mat = functools.reduce(np.kron, mats)
            coefficient = operators.trace_inner_product(mat, matrix)
            new_matrix += coefficient * mat
        if term > max_term:
            break
    return new_matrix


def get_nonzero_terms(matrix: np.ndarray, cutoff=1e-3) -> Iterator[str]:
    num_sites = int(np.round(np.log2(matrix.shape[0])))
    for mats_labels in itertools.product(zip(op_mats, ["I", "Z", "X", "Y"]), repeat=num_sites):
        mats, labels = zip(*mats_labels)
        mat = functools.reduce(np.kron, mats)
        val = operators.trace_inner_product(mat, matrix)
        if abs(val) > cutoff:
            yield "".join(labels) + " " + str(val)


def test_commutation(num_sites: int, depth: int = 3) -> None:

    test_mats: dict[str | tuple, np.ndarray] = {}
    test_ops: dict[str | tuple, operators.DenseMultiBodyOperators] = {}

    test_mats = {"a": get_random_op(num_sites), "b": get_random_op(num_sites)}

    test_ops = {
        label: operators.DenseMultiBodyOperators.from_matrix(mat, op_mats)
        for label, mat in test_mats.items()
    }

    for _ in range(depth):
        for aa_bb in itertools.combinations(test_mats.keys(), 2):
            for aa, bb in itertools.permutations(aa_bb):
                key = (aa, bb)
                if key not in test_mats.keys():
                    print(key)
                    test_mats[key] = operators.commute_mats(test_mats[aa], test_mats[bb])

                    test_ops[key] = operators.commute_dense_ops(
                        test_ops[aa],
                        test_ops[bb],
                        structure_factors
                    )

                    mat = test_mats[key]
                    op = test_ops[key]

                    success = np.allclose(op.to_matrix(op_mats), mat)
                    if not success:
                        print()
                        print("FAILURE")
                        exit()

    print("SUCCESS")


test_commutation(4)
