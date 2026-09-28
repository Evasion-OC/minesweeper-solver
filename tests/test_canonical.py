"""D4 canonical forms of regions: a region seen again, in any orientation, is not solved again."""

import random
from itertools import combinations

import pytest

import minesweeper.deductive as deductive
from minesweeper.deductive import (
    FLAGGED,
    UNKNOWN,
    InconsistentObservation,
    Observation,
    RegionCache,
    choose_action,
    deduce_all,
)
from minesweeper.symmetry import PLANE_TRANSFORMS, canonical_form, grid_transforms
from test_regions import neighbours, positions_from_play, reference_forced

U = UNKNOWN


def transformed(observation, transform):
    cells = [None] * len(observation.cells)
    for index, image in enumerate(transform.mapping):
        cells[image] = observation.cells[index]
    return Observation(observation.n_rows, observation.n_cols, observation.total_mines, tuple(cells))


def moves(actions):
    return {(action.kind, action.cell) for action in actions}


def test_canonical_form_ignores_rotation_reflection_and_translation():
    rng = random.Random(1)
    for _ in range(300):
        points = {(rng.randrange(5), rng.randrange(5)): rng.choice([U, 0, 1, 2, 3]) for _ in range(rng.randint(1, 9))}
        picture = [(row, column, label) for (row, column), label in points.items()]
        form, placements = canonical_form(picture)
        for _, transform in PLANE_TRANSFORMS:
            shift = (rng.randrange(-9, 9), rng.randrange(-9, 9))
            moved = [(*map(sum, zip(transform(row, column), shift)), label) for row, column, label in picture]
            assert canonical_form(moved)[0] == form
        for placement in placements:  # every placement lays the picture exactly onto the form
            assert tuple(sorted((*placement[(row, column)], label) for row, column, label in picture)) == form


def test_the_placements_show_a_pictures_own_symmetries():
    # A 1 above three covered cells is symmetric under the reflection that swaps the outer two.
    t_shape = [(0, 1, 1), (1, 0, U), (1, 1, U), (1, 2, U)]
    _, placements = canonical_form(t_shape)
    assert len(placements) == 2
    orbits = {frozenset(placement[cell] for placement in placements) for cell in [(1, 0), (1, 1), (1, 2)]}
    assert sorted(map(len, orbits)) == [1, 2]
    # An L of a clue and two covered cells is symmetric only under the diagonal reflection.
    assert len(canonical_form([(0, 0, 2), (0, 1, U), (1, 0, U)])[1]) == 2
    # With a different clue beside each covered cell nothing is swapped.
    assert len(canonical_form([(0, 0, 1), (0, 1, U), (1, 1, 2)])[1]) == 1


@pytest.mark.parametrize(("n_rows", "n_cols", "num_mines", "games"), [(6, 6, 8, 60), (8, 8, 10, 40), (9, 9, 10, 25)])
def test_cached_deductions_equal_fresh_ones(n_rows, n_cols, num_mines, games):
    cache = RegionCache()  # kept across every position and game
    for observation in positions_from_play(n_rows, n_cols, num_mines, games, seed=n_rows * 7 + num_mines):
        cached = deduce_all(observation, cache)
        assert cached == deduce_all(observation)
        assert moves(cached) == reference_forced(observation)
        assert choose_action(observation, cache) == choose_action(observation)
    assert cache.hits > cache.misses > 0


def test_a_rotated_or_reflected_board_is_answered_from_the_cache():
    for observation in positions_from_play(9, 9, 10, games=4, seed=11):
        cache = RegionCache()
        original = moves(deduce_all(observation, cache))
        solved = cache.misses
        for transform in grid_transforms(9, 9):
            answer = moves(deduce_all(transformed(observation, transform), cache))
            assert answer == {(kind, transform(*cell)) for kind, cell in original}
        assert cache.misses == solved  # no region of the seven other boards was solved again


def test_a_symmetric_region_tests_one_cell_per_orbit(monkeypatch):
    # A 1-2-1 against the wall is symmetric under the left-right reflection, so the
    # two outer cells form one orbit: two cells are tested instead of three.
    tested = []
    real = deductive._forced_kind
    monkeypatch.setattr(deductive, "_forced_kind", lambda solver, variable: tested.append(variable) or real(solver, variable))
    wall = Observation(2, 3, 2, (U, U, U, 1, 2, 1))
    assert moves(deduce_all(wall)) == {("flag", (0, 0)), ("reveal", (0, 1)), ("flag", (0, 2))}
    assert len(tested) == 2


def some_layout_fits(observation):
    """Whether any placement of the remaining mines satisfies every clue (flags count as mines)."""
    n_rows, n_cols, cells = observation.n_rows, observation.n_cols, observation.cells
    unknown = [i for i, s in enumerate(cells) if s == UNKNOWN]
    remaining = observation.total_mines - cells.count(FLAGGED)
    if not 0 <= remaining <= len(unknown):
        return False
    for layout in combinations(unknown, remaining):
        mines = set(layout) | {i for i, s in enumerate(cells) if s == FLAGGED}
        if all(sum(n in mines for n in neighbours(i, n_rows, n_cols)) == s for i, s in enumerate(cells) if s >= 0):
            return True
    return False


def test_a_board_no_layout_fits_is_always_rejected():
    # The whole board is built only when the mine count may bind, so rejection rests on
    # the checks of each clue, each region and the count; this compares it with enumeration.
    rng = random.Random(5)
    shapes = [(1, 3), (2, 2), (2, 3), (3, 3), (1, 5), (2, 4)]
    shared = RegionCache()  # also kept across every board: a region no layout fits must never be stored
    rejected = accepted = 0
    for trial in range(3000):
        n_rows, n_cols = rng.choice(shapes)
        size = n_rows * n_cols
        if trial % 2:  # a random visible state, rarely consistent
            cells = [rng.choice([U, U, U, FLAGGED, 0, 1, 1, 2, 3]) for _ in range(size)]
            total = rng.randrange(size + 1)
        else:  # seen from a real layout, then sometimes one cell or the count changed
            mines = set(rng.sample(range(size), rng.randrange(size)))
            cells = [rng.choice([U, FLAGGED]) if i in mines else
                     rng.choice([U, sum(n in mines for n in neighbours(i, n_rows, n_cols))]) for i in range(size)]
            total = len(mines)
            if rng.random() < 0.4:
                cells[rng.randrange(size)] = rng.choice([U, FLAGGED, 0, 1, 2])
            if rng.random() < 0.2:
                total = rng.randrange(size + 1)
        observation = Observation(n_rows, n_cols, total, tuple(cells))
        if some_layout_fits(observation):
            accepted += 1
            assert moves(deduce_all(observation)) == moves(deduce_all(observation, shared)) == reference_forced(observation)
            assert choose_action(observation, shared) == choose_action(observation)
        else:
            rejected += 1
            for cache in (None, shared, shared):
                with pytest.raises(InconsistentObservation):
                    deduce_all(observation, cache)
                with pytest.raises(InconsistentObservation):
                    choose_action(observation, cache)
    assert rejected > 500 and accepted > 300


def test_the_whole_board_is_built_only_when_the_count_may_bind(monkeypatch):
    built = []
    real = deductive._constraint_model
    monkeypatch.setattr(deductive, "_constraint_model", lambda observation: built.append(1) or real(observation))
    # One region, 1 or 2 mines, three cells left for them: the count cannot bind.
    assert moves(deduce_all(Observation(1, 6, 2, (U, 1, U, 1, U, U)))) == set()
    assert built == []
    # The same region with nothing else covered and one mine: the count decides it.
    assert moves(deduce_all(Observation(1, 5, 1, (U, 1, U, 1, U)))) == {
        ("reveal", (0, 0)), ("flag", (0, 2)), ("reveal", (0, 4))}
    assert built == [1]
    # Positions from play: the whole board is built for few of them.
    positions = list(positions_from_play(9, 9, 10, games=10, seed=21))
    built.clear()
    cache = RegionCache()
    for observation in positions:
        choose_action(observation, cache)
    assert len(built) < len(positions) / 5
