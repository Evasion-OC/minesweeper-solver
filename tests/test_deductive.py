from itertools import combinations

import pytest

from minesweeper.board import compute_clues, create_board
from minesweeper.deductive import (
    FLAGGED,
    UNKNOWN,
    Action,
    InconsistentObservation,
    Observation,
    choose_action,
    deduce_all,
)
from minesweeper.symmetry import cell_orbits, grid_transforms, state_stabilizer


@pytest.mark.parametrize(
    ("n_rows", "n_cols", "order"),
    [(1, 1, 1), (1, 3, 2), (2, 2, 8), (2, 3, 4), (3, 3, 8)],
)
def test_shape_preserving_group(n_rows, n_cols, order):
    transforms = grid_transforms(n_rows, n_cols)
    mappings = {transform.mapping for transform in transforms}
    identity = tuple(range(n_rows * n_cols))
    assert len(transforms) == len(mappings) == order
    assert identity in mappings
    assert all(set(mapping) == set(identity) for mapping in mappings)
    for left in transforms:
        for right in transforms:
            composed = tuple(left.mapping[right.mapping[index]] for index in identity)
            assert composed in mappings


def test_only_visible_state_symmetries_reduce_deduction():
    fully_covered = (UNKNOWN,) * 9
    assert len(state_stabilizer(fully_covered, 3, 3)) == 8
    asymmetric = (FLAGGED, 1, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN)
    assert len(state_stabilizer(asymmetric, 3, 3)) == 1
    transforms = state_stabilizer(fully_covered, 3, 3)
    assert cell_orbits(range(9), transforms) == ((0, 2, 6, 8), (1, 3, 5, 7), (4,))
    # A board kept only by the half turn: the two reflections of the 2x3 grid would copy the
    # safe cell (0, 0) onto the clues at (0, 2) and (1, 0). The one mine lies in both 1s'
    # neighbourhoods, at (0, 1) or (1, 1), so (0, 0) and (1, 2) are safe and nothing else is decided.
    observation = Observation(2, 3, 1, (UNKNOWN, UNKNOWN, 1, 1, UNKNOWN, UNKNOWN))
    assert len(state_stabilizer(observation.cells, 2, 3)) == 2 < len(grid_transforms(2, 3))
    assert {(action.kind, action.cell) for action in deduce_all(observation)} == {("reveal", (0, 0)), ("reveal", (1, 2))}


def test_hidden_layout_and_covered_clues_cannot_change_action():
    first = [[
        {"covered": False, "flagged": False, "isMine": False, "clue": 1},
        {"covered": True, "flagged": False, "isMine": True, "clue": -1},
    ], [
        {"covered": True, "flagged": False, "isMine": False, "clue": 1},
        {"covered": True, "flagged": False, "isMine": False, "clue": 1},
    ]]
    second = [[
        {"covered": False, "flagged": False, "isMine": False, "clue": 1},
        {"covered": True, "flagged": False, "isMine": False, "clue": 1},
    ], [
        {"covered": True, "flagged": False, "isMine": True, "clue": -1},
        {"covered": True, "flagged": False, "isMine": False, "clue": 1},
    ]]
    first_observation = Observation.from_board(first, 1)
    second_observation = Observation.from_board(second, 1)
    assert first_observation == second_observation
    assert choose_action(first_observation) == choose_action(second_observation)
    assert deduce_all(first_observation) == deduce_all(second_observation)


def test_covered_cells_do_not_need_hidden_fields():
    board = [[{"covered": True, "flagged": False}]]
    observation = Observation.from_board(board, 0)
    assert choose_action(observation).reason == "forced"


def test_revealed_cells_use_only_valid_visible_clues():
    board = [[{"covered": False, "flagged": False, "clue": 0}]]
    assert Observation.from_board(board, 0).cells == (0,)
    with pytest.raises(InconsistentObservation):
        Observation.from_board(
            [[{"covered": False, "flagged": False, "clue": -1, "isMine": True}]],
            0,
        )


def test_forced_actions_transform_with_visible_board():
    observation = Observation(2, 3, 1, (1, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN))
    original = {
        (action.kind, action.cell[0] * observation.n_cols + action.cell[1])
        for action in deduce_all(observation)
    }
    for transform in grid_transforms(observation.n_rows, observation.n_cols):
        transformed_cells = [None] * len(observation.cells)
        for index, state in enumerate(observation.cells):
            transformed_cells[transform.mapping[index]] = state
        transformed_observation = Observation(
            observation.n_rows, observation.n_cols, observation.total_mines, tuple(transformed_cells)
        )
        transformed_actions = {
            (action.kind, action.cell[0] * observation.n_cols + action.cell[1])
            for action in deduce_all(transformed_observation)
        }
        assert transformed_actions == {
            (kind, transform.mapping[index]) for kind, index in original
        }


def test_endgame_counterexample_is_a_guess_not_a_proof():
    observation = Observation(1, 3, 2, (FLAGGED, UNKNOWN, UNKNOWN))
    assert deduce_all(observation) == ()
    assert choose_action(observation).reason == "guess"


def test_total_mine_count_proves_cells_outside_visible_clues():
    observation = Observation(1, 3, 1, (FLAGGED, UNKNOWN, UNKNOWN))
    assert deduce_all(observation) == (
        Action("reveal", (0, 1), "forced"),
        Action("reveal", (0, 2), "forced"),
    )


@pytest.mark.parametrize(("n_rows", "n_cols", "num_mines"), [(1, 3, 2), (2, 2, 1), (2, 3, 2)])
def test_exhaustive_tiny_boards_have_no_false_proofs(n_rows, n_cols, num_mines):
    cells = tuple((row, column) for row in range(n_rows) for column in range(n_cols))
    for mine_cells in combinations(cells, num_mines):
        mines = set(mine_cells)
        grid = [["*" if (row, column) in mines else "." for column in range(n_cols)] for row in range(n_rows)]
        compute_clues(grid)
        safe_cells = tuple(cell for cell in cells if cell not in mines)
        for num_revealed in range(len(safe_cells) + 1):
            for revealed in combinations(safe_cells, num_revealed):
                for num_flagged in range(len(mines) + 1):
                    for flagged in combinations(mine_cells, num_flagged):
                        board = create_board(grid)
                        for row, column in revealed:
                            board[row][column]["covered"] = False
                        for row, column in flagged:
                            board[row][column]["flagged"] = True
                        observation = Observation.from_board(board, num_mines)
                        for action in deduce_all(observation):
                            assert (action.cell in mines) == (action.kind == "flag")
                            assert action.reason == "forced"


def test_inconsistent_observation_stops_instead_of_guessing():
    with pytest.raises(InconsistentObservation):
        choose_action(Observation(1, 2, 0, (FLAGGED, UNKNOWN)))
    with pytest.raises(InconsistentObservation):
        choose_action(Observation(1, 2, 1, (0, FLAGGED)))
