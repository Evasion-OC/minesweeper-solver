"""The play pipeline: single clue, pattern, SAT, guess, with every non-guess move forced."""

import pytest

from itertools import combinations

import minesweeper.pipeline as pipeline
from minesweeper.deductive import FLAGGED, UNKNOWN, InconsistentObservation, Observation, choose_action, deduce_all
from minesweeper.patterns import Proposal, propose
from minesweeper.pipeline import STAGES, PlaySolver, single_clue_moves
from test_regions import neighbours, positions_from_play, reference_forced

U = UNKNOWN


@pytest.mark.parametrize(("board", "stage", "move"), [
    # The 1 sees one covered cell: one clue decides it.
    (Observation(1, 3, 1, (U, 1, 0)), "single clue", ("flag", (0, 0))),
    # 1-2-1 against the wall: no clue decides anything alone, the 1_2_1 pattern does.
    (Observation(2, 3, 2, (U, U, U, 1, 2, 1)), "pattern", None),
    # Two 1s two apart with one mine in all: only the mine count decides it.
    (Observation(1, 5, 1, (U, 1, U, 1, U)), "SAT", ("reveal", (0, 0))),
    # One 1 and two cells, nothing forced.
    (Observation(1, 3, 1, (U, 1, U)), "guess", ("reveal", (0, 0))),
])
def test_each_stage_is_reached_in_order(board, stage, move):
    solver = PlaySolver()
    step = solver.step(board)
    assert step.stage == stage
    if move is not None:
        assert (step.action.kind, step.action.cell) == move
    assert solver.stage_counts == {stage: 1}


def test_single_clue_moves():
    board = Observation(2, 4, 2, (U, U, U, U, 0, 0, 2, U))
    # The 0s clear their covered neighbours; the 2 at (1, 2) sees four covered cells, so nothing from it.
    assert single_clue_moves(board) == [("reveal", 0), ("reveal", 1), ("reveal", 2)]
    # Flags count: the 1 beside a flag has no mine left, the 2 beside a flag has one left for one cell.
    assert single_clue_moves(Observation(1, 3, 1, (FLAGGED, 1, U))) == [("reveal", 2)]
    assert single_clue_moves(Observation(1, 3, 2, (FLAGGED, 2, U))) == [("flag", 2)]


def decided_by_one_clue(observation):
    """Moves some clue forces on its own: every placement of its missing mines agrees."""
    n_rows, n_cols, cells = observation.n_rows, observation.n_cols, observation.cells
    moves = set()
    for index, state in enumerate(cells):
        if state < 0:
            continue
        around = neighbours(index, n_rows, n_cols)
        covered = [n for n in around if cells[n] == UNKNOWN]
        placements = [set(p) for p in combinations(covered, state - sum(cells[n] == FLAGGED for n in around))]
        for cell in covered:
            if all(cell in p for p in placements):
                moves.add(("flag", cell))
            elif not any(cell in p for p in placements):
                moves.add(("reveal", cell))
    return moves


def test_single_clue_moves_match_an_independent_check_in_play():
    checked = flagged = 0
    for observation in positions_from_play(8, 8, 12, games=40, seed=6):  # the solver's own flags included
        assert set(single_clue_moves(observation)) == decided_by_one_clue(observation)
        checked += 1
        flagged += FLAGGED in observation.cells
    assert checked > 500 and flagged > 200


def test_a_rule_that_sat_does_not_confirm_is_not_played(monkeypatch):
    # The count forces flag (0, 2) and reveal (0, 0) and (0, 4); a wrong claim must be passed over.
    board = Observation(1, 5, 1, (U, 1, U, 1, U))
    monkeypatch.setattr(pipeline, "single_clue_moves", lambda observation: [("flag", 0)])
    monkeypatch.setattr(pipeline, "propose", lambda observation: (Proposal("wrong", "flag", (0, 4)),))
    step = PlaySolver().step(board)
    assert (step.action.kind, step.action.cell, step.stage) == ("reveal", (0, 0), "SAT")


def test_a_new_game_forgets_the_regions_and_keeps_the_counts():
    solver = PlaySolver()
    solver.step(Observation(2, 3, 2, (U, U, U, 1, 2, 1)))
    assert len(solver.cache) > 0 and solver.stage_counts == {"pattern": 1}
    solver.new_game()
    assert len(solver.cache) == 0 and solver.stage_counts == {"pattern": 1}


def test_every_move_before_a_guess_is_forced_and_the_guess_is_the_cores():
    for n_rows, n_cols, num_mines, games in [(6, 6, 7, 40), (9, 9, 10, 20), (8, 8, 12, 30)]:
        solver = PlaySolver()  # one solver, and one cache of solved regions, across all games
        for observation in positions_from_play(n_rows, n_cols, num_mines, games, seed=n_rows + num_mines):
            step = solver.step(observation)
            forced = reference_forced(observation)
            if forced:
                assert step.stage != "guess"
                assert (step.action.kind, step.action.cell) in forced
            else:
                assert step.stage == "guess"
                assert step.action == choose_action(observation)
        assert set(solver.stage_counts) <= set(STAGES)
        assert solver.stage_counts["single clue"] > 0 and solver.stage_counts["guess"] > 0


def test_the_stage_is_the_earliest_that_finds_a_move():
    solver = PlaySolver()
    for observation in positions_from_play(8, 8, 12, games=20, seed=5):
        step = solver.step(observation)
        forced = {(a.kind, a.cell[0] * observation.n_cols + a.cell[1]) for a in deduce_all(observation)}
        if step.stage in ("pattern", "SAT"):
            assert not forced & set(single_clue_moves(observation))
        if step.stage == "SAT":
            assert not forced & {(p.kind, p.cell[0] * observation.n_cols + p.cell[1]) for p in propose(observation)}


def test_an_inconsistent_board_stops_the_solver():
    with pytest.raises(InconsistentObservation):
        PlaySolver().step(Observation(1, 3, 0, (1, U, 1)))
