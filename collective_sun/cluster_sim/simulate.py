#!/usr/bin/env python3
import itertools
from typing import Sequence

import numpy as np

import operators
import equations


def get_location(position: Sequence[int], axis_sizes: Sequence[int]) -> int:
    """Return the lattice site corresponding to a given position."""
    return int(np.ravel_multi_index(position, axis_sizes, mode="wrap"))


def get_position(site: int, axis_sizes: Sequence[int]) -> tuple[int, ...]:
    """Get the physical position of a given lattice site."""
    return tuple(int(ii) for ii in np.unravel_index(site, axis_sizes))


def get_distance(site_a: int, site_b: int) -> float:
    """Get the physical distance between two lattice sites."""
    loc_a = get_position(site_a, axis_sizes)
    loc_b = get_position(site_b, axis_sizes)
    return np.sqrt(sum(abs(aa - bb) ** 2 for aa, bb in zip(loc_a, loc_b)))


def get_coupling(
    site_a: int,
    site_b: int,
    axis_sizes: Sequence[int],
    blockade_radius: float = 2.0,
    exponent: float = 6.0,
) -> float:
    """Get the coupling strength between spins at the given lattice sites."""
    if site_a == site_b:
        return 0
    return 1 / (1 + (get_distance(site_a, site_b) / blockade_radius) ** exponent)


if __name__ == "__main__":

    axis_sizes = (6, 6)
    num_sites = np.prod(axis_sizes, dtype=int)

    cluster_radius = 1

    # identify qubit operators
    op_I, op_Z, op_X, op_Y = operators.AbstractSingleBodyOperator.range(4)
    qubit_op_mats = operators.get_qudit_op_mats(2)
    structure_factors = operators.get_structure_factors(*qubit_op_mats)

    # build the hamiltonian
    coupling_mat = 0.25 * np.array(
        [get_coupling(aa, bb, axis_sizes) for aa in range(num_sites) for bb in range(num_sites)]
    ).reshape((num_sites, num_sites))
    hamiltonian = operators.DenseMultiBodyOperator(op_Z, op_Z, tensor=coupling_mat)

    # identify factorization rule
    def keep_rule(op: operators.MultiBodyOperator):
        return not any(
            get_distance(op_a.site.index, op_b.site.index) > 2 * cluster_radius
            for op_a, op_b in itertools.combinations(op.ops, 2)
        )

    # build equations of motion
    op_to_index, time_deriv_tensors = equations.build_equations_of_motion(
        hamiltonian=hamiltonian,
        structure_factors=structure_factors,
        factorization_rule=equations.get_cumulant_factorizer(keep_rule),
        show_progress=True,
    )
