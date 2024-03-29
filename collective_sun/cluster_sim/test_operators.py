import functools
import itertools
from typing import Iterator

import numpy as np
import pytest

import operators

np.random.seed(0)
np.set_printoptions(linewidth=200, precision=3)

QUBIT_OP_MATS = operators.get_qubit_op_mats()
STRUCTURE_FACTORS = operators.get_structure_factors(*QUBIT_OP_MATS)


def get_nonzero_terms(matrix: np.ndarray, cutoff=1e-3) -> Iterator[str]:
    num_sites = int(np.round(np.log2(matrix.shape[0])))
    for mats_labels in itertools.product(
        zip(QUBIT_OP_MATS, ["I", "Z", "X", "Y"]), repeat=num_sites
    ):
        mats, labels = zip(*mats_labels)
        mat = functools.reduce(np.kron, mats)
        val = operators.trace_inner_product(mat, matrix)
        if abs(val) > cutoff:
            yield "".join(labels) + " " + str(val)


@pytest.mark.parametrize("num_sites,depth", [(3, 3)])
def test_commutation(num_sites: int, depth: int) -> None:
    test_mats: dict[str | tuple, np.ndarray] = {}
    test_ops: dict[str | tuple, operators.DenseMultiBodyOperators] = {}

    test_mats = {
        "a": operators.get_random_matrix(2**num_sites),
        "b": operators.get_random_matrix(2**num_sites),
    }
    test_ops = {
        label: operators.DenseMultiBodyOperators.from_matrix(mat, QUBIT_OP_MATS)
        for label, mat in test_mats.items()
    }

    print()
    for _ in range(depth):
        for aa_bb in itertools.combinations(test_mats.keys(), 2):
            for aa, bb in itertools.permutations(aa_bb):
                key = (aa, bb)
                if key not in test_mats.keys():
                    print(key)
                    test_mats[key] = operators.commute_mats(test_mats[aa], test_mats[bb])
                    test_ops[key] = operators.commute_dense_ops(
                        test_ops[aa], test_ops[bb], STRUCTURE_FACTORS
                    )
                    success = np.allclose(test_ops[key].to_matrix(QUBIT_OP_MATS), test_mats[key])
                    assert success
