"""The solver's play pipeline: the simplest rule that justifies a move, every move checked by SAT.

Each step first finds every forced move with the SAT core (region by region, with the
cache of solved regions), which also rejects a board no mine layout fits. It then plays
the move found by the earliest of these stages:

1. single clue: a clue whose mines are all found, so its other covered neighbours are
   safe, or whose covered neighbours must all be mines;
2. pattern: a move from the pattern library (minesweeper.patterns);
3. SAT: any other forced move, which needs several clues or the mine count together;
4. guess: when nothing is forced, the covered cell least likely to be a mine.

A single-clue or pattern move is played only if SAT also finds it forced. The counts
per stage record the simplest rule that justifies each move.
"""

from collections import Counter
from dataclasses import dataclass

from minesweeper.deductive import (
    FLAGGED,
    UNKNOWN,
    Action,
    Observation,
    RegionCache,
    _neighbours,
    choose_action,
    deduce_all,
)
from minesweeper.patterns import propose

STAGES = ("single clue", "pattern", "SAT", "guess")


def single_clue_moves(observation: Observation) -> list[tuple[str, int]]:
    """Moves one clue decides on its own, as (kind, cell index) in board order."""
    n_rows, n_cols, cells = observation.n_rows, observation.n_cols, observation.cells
    moves = {}
    for index, state in enumerate(cells):
        if state < 0:
            continue
        around = _neighbours(index, n_rows, n_cols)
        covered = [n for n in around if cells[n] == UNKNOWN]
        left = state - sum(cells[n] == FLAGGED for n in around)
        if covered and left == 0:
            moves.update((cell, "reveal") for cell in covered)
        elif covered and left == len(covered):
            moves.update((cell, "flag") for cell in covered)
    return [(kind, cell) for cell, kind in sorted(moves.items())]


@dataclass(frozen=True)
class Step:
    action: Action
    stage: str


class PlaySolver:
    """Chooses one move at a time. It keeps the solved regions of the current game and
    counts the moves per stage across games."""

    def __init__(self) -> None:
        self.cache = RegionCache()
        self.stage_counts = Counter()

    def new_game(self) -> None:
        """Forget the solved regions, which keeps memory bounded; keep the counts."""
        self.cache = RegionCache()

    def step(self, observation: Observation) -> Step | None:
        """The next move and the stage that found it, or None when nothing is covered.

        Raises InconsistentObservation when no mine layout fits the board.
        """
        n_cols = observation.n_cols
        forced = {(action.kind, action.cell[0] * n_cols + action.cell[1]): action
                  for action in deduce_all(observation, self.cache)}
        step = None
        if forced:
            for move in single_clue_moves(observation):
                if move in forced:
                    step = Step(forced[move], "single clue")
                    break
            if step is None:
                for proposal in propose(observation):
                    move = (proposal.kind, proposal.cell[0] * n_cols + proposal.cell[1])
                    if move in forced:
                        step = Step(forced[move], "pattern")
                        break
            if step is None:
                step = Step(min(forced.values(), key=lambda action: action.cell), "SAT")
        else:
            action = choose_action(observation, self.cache)
            if action is None:
                return None
            step = Step(action, "guess")
        self.stage_counts[step.stage] += 1
        return step
