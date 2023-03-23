import numpy as np

import simulate


def test_placements() -> None:
    axis_sizes = (10, 10)
    num_sites = np.prod(axis_sizes, dtype=int)

    for pos in np.ndindex(axis_sizes):
        loc = simulate.get_location(pos, axis_sizes)
        assert pos == simulate.get_position(loc, axis_sizes)

    for loc in range(num_sites):
        pos = simulate.get_position(loc, axis_sizes)
        assert loc == simulate.get_location(pos, axis_sizes)
