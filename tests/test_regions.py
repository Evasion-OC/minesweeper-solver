"""Independent regions: the frontier splits into parts that are solved separately."""

import random

import pytest
from pysat.card import CardEnc, EncType
from pysat.formula import IDPool
from pysat.solvers import Glucose3

from minesweeper.deductive import (
    FLAGGED,
    UNKNOWN,
    Action,
    Observation,
    choose_action,
    deduce_all,
    frontier_regions,
)


def neighbours(index, n_rows, n_cols):
    row, column = divmod(index, n_cols)
    return [
        r * n_cols + c
        for r in range(max(0, row - 1), min(n_rows, row + 2))
        for c in range(max(0, column - 1), min(n_cols, column + 2))
        if (r, c) != (row, column)
    ]


def clues_for(mines, n_rows, n_cols):
    return [sum(n in mines for n in neighbours(i, n_rows, n_cols)) for i in range(n_rows * n_cols)]


def reference_forced(observation):
    """A cell is forced when one of its two values contradicts all clues and the mine count."""
    n_rows, n_cols, cells = observation.n_rows, observation.n_cols, observation.cells
    unknown = [i for i, s in enumerate(cells) if s == UNKNOWN]
    var = {i: k + 1 for k, i in enumerate(unknown)}
    pool = IDPool(start_from=len(unknown) + 1)
    clauses = CardEnc.equals(lits=list(var.values()), bound=observation.total_mines - cells.count(FLAGGED),
                             vpool=pool, encoding=EncType.totalizer).clauses
    for i, s in enumerate(cells):
        if s >= 0:
            around = neighbours(i, n_rows, n_cols)
            clauses += CardEnc.equals(lits=[var[n] for n in around if cells[n] == UNKNOWN],
                                      bound=s - sum(cells[n] == FLAGGED for n in around),
                                      vpool=pool, encoding=EncType.totalizer).clauses
    solver = Glucose3(bootstrap_with=clauses)
    forced = set()
    for i, v in var.items():
        if not solver.solve(assumptions=[v]):
            forced.add(("reveal", divmod(i, n_cols)))
        elif not solver.solve(assumptions=[-v]):
            forced.add(("flag", divmod(i, n_cols)))
    solver.delete()
    return forced


def positions_from_play(n_rows, n_cols, num_mines, games, seed):
    """Every position met while the solver plays, including late ones where the count matters."""
    rng = random.Random(seed)
    for _ in range(games):
        first = rng.randrange(n_rows * n_cols)
        banned = set(neighbours(first, n_rows, n_cols)) | {first}
        mines = set(rng.sample([i for i in range(n_rows * n_cols) if i not in banned], num_mines))
        clue = clues_for(mines, n_rows, n_cols)
        visible = [UNKNOWN] * (n_rows * n_cols)

        def reveal(i):
            stack = [i]
            while stack:
                j = stack.pop()
                if visible[j] != UNKNOWN:
                    continue
                visible[j] = clue[j]
                if clue[j] == 0:
                    stack.extend(neighbours(j, n_rows, n_cols))

        reveal(first)
        while True:
            observation = Observation(n_rows, n_cols, num_mines, tuple(visible))
            if all(visible[i] != UNKNOWN for i in range(n_rows * n_cols) if i not in mines):
                break
            yield observation
            action = choose_action(observation)
            index = action.cell[0] * n_cols + action.cell[1]
            if action.kind == "flag":
                visible[index] = FLAGGED
            elif index in mines:
                break
            else:
                reveal(index)


def test_regions_partition_the_frontier():
    for observation in positions_from_play(9, 9, 10, games=30, seed=1):
        cells, n_rows, n_cols = observation.cells, observation.n_rows, observation.n_cols
        regions = frontier_regions(observation)
        frontier = {n for i, s in enumerate(cells) if s >= 0
                    for n in neighbours(i, n_rows, n_cols) if cells[n] == UNKNOWN}
        region_cells = [set(region.cells) for region in regions]
        assert set().union(*region_cells) == frontier
        assert sum(map(len, region_cells)) == len(frontier)  # disjoint
        for region in regions:
            for clue in region.clues:
                touched = {n for n in neighbours(clue, n_rows, n_cols) if cells[n] == UNKNOWN}
                assert touched and touched <= set(region.cells)


def test_a_region_is_unaffected_by_other_regions():
    # Two revealed bands (columns 0-2 and 10-12) with covered, clue-free columns between.
    n_rows, n_cols = 6, 13
    rng = random.Random(7)
    for trial in range(40):
        left = set(rng.sample([r * n_cols + c for r in range(n_rows) for c in range(3, 5)], 3))
        right_a = set(rng.sample([r * n_cols + c for r in range(n_rows) for c in range(8, 10)], 3))
        right_b = set(rng.sample([r * n_cols + c for r in range(n_rows) for c in range(8, 10)], 3))
        middle = {r * n_cols + 6 for r in range(n_rows)}
        results = []
        for mines in (left | right_a | middle, left | right_b | middle):
            clue = clues_for(mines, n_rows, n_cols)
            visible = [clue[i] if i % n_cols in (0, 1, 2, 10, 11, 12) and i not in mines else UNKNOWN
                       for i in range(n_rows * n_cols)]
            observation = Observation(n_rows, n_cols, len(mines), tuple(visible))
            results.append({(a.kind, a.cell) for a in deduce_all(observation) if a.cell[1] <= 3})
        assert results[0] == results[1]


@pytest.mark.parametrize(("n_rows", "n_cols", "num_mines", "games"),
                         [(5, 5, 5, 60), (6, 6, 8, 60), (8, 8, 10, 40), (9, 9, 10, 25)])
def test_deductions_match_the_whole_board_reference(n_rows, n_cols, num_mines, games):
    checked = 0
    for observation in positions_from_play(n_rows, n_cols, num_mines, games, seed=n_rows * 100 + num_mines):
        found = {(a.kind, a.cell) for a in deduce_all(observation)}
        assert found == reference_forced(observation)
        checked += 1
    assert checked > 100


def test_the_mine_count_still_decides_cells_no_clue_touches():
    # The clue sees one cell, which must be the only mine; the count then clears the rest.
    observation = Observation(1, 4, 1, (1, UNKNOWN, UNKNOWN, UNKNOWN))
    assert deduce_all(observation) == (
        Action("flag", (0, 1), "forced"),
        Action("reveal", (0, 2), "forced"),
        Action("reveal", (0, 3), "forced"),
    )
