from itertools import combinations

import pytest

from minesweeper.board import compute_clues, create_board
from minesweeper.deductive import FLAGGED, UNKNOWN, Observation, deduce_all


def _neighbours(index, n_rows, n_cols):
    row, column = divmod(index, n_cols)
    return {
        next_row * n_cols + next_column
        for next_row in range(max(0, row - 1), min(n_rows, row + 2))
        for next_column in range(max(0, column - 1), min(n_cols, column + 2))
        if (next_row, next_column) != (row, column)
    }


def _oracle_forced_actions(observation):
    cell_count = observation.n_rows * observation.n_cols
    placements = []
    for mine_indices in combinations(range(cell_count), observation.total_mines):
        mines = set(mine_indices)
        if any(
            state == FLAGGED and index not in mines
            for index, state in enumerate(observation.cells)
        ):
            continue
        if any(
            state >= 0 and index in mines
            for index, state in enumerate(observation.cells)
        ):
            continue
        consistent = True
        for index, clue in enumerate(observation.cells):
            if clue < 0:
                continue
            if sum(neighbour in mines for neighbour in _neighbours(
                index, observation.n_rows, observation.n_cols
            )) != clue:
                consistent = False
                break
        if consistent:
            placements.append(mines)

    assert placements, "the generated observation must be consistent"
    expected = set()
    for index, state in enumerate(observation.cells):
        if state != UNKNOWN:
            continue
        values = {index in mines for mines in placements}
        if values == {False}:
            expected.add((index, "reveal"))
        elif values == {True}:
            expected.add((index, "flag"))
    return expected


@pytest.mark.parametrize(
    ("n_rows", "n_cols", "mine_count"),
    [(1, 3, 0), (1, 3, 1), (1, 3, 2), (2, 2, 1)],
)
def test_deductions_equal_independent_bruteforce_oracle(n_rows, n_cols, mine_count):
    cell_indices = tuple(range(n_rows * n_cols))
    for mine_indices in combinations(cell_indices, mine_count):
        mine_set = set(mine_indices)
        grid = [
            [
                "*" if row * n_cols + column in mine_set else "."
                for column in range(n_cols)
            ]
            for row in range(n_rows)
        ]
        compute_clues(grid)
        safe_indices = tuple(index for index in cell_indices if index not in mine_set)

        for reveal_count in range(len(safe_indices) + 1):
            for revealed in combinations(safe_indices, reveal_count):
                for flag_count in range(mine_count + 1):
                    for flagged in combinations(mine_indices, flag_count):
                        board = create_board(grid)
                        for index in revealed:
                            row, column = divmod(index, n_cols)
                            board[row][column]["covered"] = False
                        for index in flagged:
                            row, column = divmod(index, n_cols)
                            board[row][column]["flagged"] = True

                        observation = Observation.from_board(board, mine_count)
                        expected = _oracle_forced_actions(observation)
                        actual = {
                            (
                                action.cell[0] * n_cols + action.cell[1],
                                action.kind,
                            )
                            for action in deduce_all(observation)
                        }
                        assert actual == expected
