# The SAT baseline for scripts/deductive_measure.py, from an earlier 2026 version of
# minesweeper/deductive.py. It solves the whole board by SAT at every move and guesses the
# covered cell with the lowest index.
from dataclasses import dataclass
from typing import Literal

from pysat.card import CardEnc, EncType
from pysat.formula import IDPool
from pysat.solvers import Glucose3

from minesweeper.symmetry import cell_orbits, state_stabilizer


UNKNOWN = -1
FLAGGED = -2
Coordinate = tuple[int, int]


class InconsistentObservation(ValueError):
    pass


@dataclass(frozen=True)
class Observation:
    n_rows: int
    n_cols: int
    total_mines: int
    cells: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.n_rows < 1 or self.n_cols < 1:
            raise InconsistentObservation("Board dimensions must be positive")
        if len(self.cells) != self.n_rows * self.n_cols:
            raise InconsistentObservation("Visible state does not match board dimensions")
        if not 0 <= self.total_mines <= len(self.cells):
            raise InconsistentObservation("Total mine count is outside the board")
        if any(state not in (UNKNOWN, FLAGGED) and not 0 <= state <= 8 for state in self.cells):
            raise InconsistentObservation("Visible state contains an invalid clue")

    @classmethod
    def from_board(cls, board: list[list[dict]], total_mines: int) -> "Observation":
        if not board or not board[0] or any(len(row) != len(board[0]) for row in board):
            raise InconsistentObservation("Board must be a non-empty rectangle")
        visible = []
        for row in board:
            for cell in row:
                if cell["flagged"]:
                    if not cell["covered"]:
                        raise InconsistentObservation("A revealed cell cannot be flagged")
                    visible.append(FLAGGED)
                elif cell["covered"]:
                    visible.append(UNKNOWN)
                else:
                    clue = cell["clue"]
                    if not 0 <= clue <= 8:
                        raise InconsistentObservation("A revealed cell has an invalid clue")
                    visible.append(clue)
        return cls(len(board), len(board[0]), total_mines, tuple(visible))


@dataclass(frozen=True)
class Action:
    kind: Literal["reveal", "flag"]
    cell: Coordinate
    reason: Literal["forced", "guess"]


def _neighbours(index: int, n_rows: int, n_cols: int) -> tuple[int, ...]:
    row, column = divmod(index, n_cols)
    return tuple(
        next_row * n_cols + next_column
        for next_row in range(max(0, row - 1), min(n_rows, row + 2))
        for next_column in range(max(0, column - 1), min(n_cols, column + 2))
        if (next_row, next_column) != (row, column)
    )


def _constraint_model(observation: Observation) -> tuple[Glucose3, dict[int, int]]:
    unknowns = [index for index, state in enumerate(observation.cells) if state == UNKNOWN]
    variables = {index: position + 1 for position, index in enumerate(unknowns)}
    flagged_count = observation.cells.count(FLAGGED)
    remaining_mines = observation.total_mines - flagged_count
    if not 0 <= remaining_mines <= len(unknowns):
        raise InconsistentObservation("Flags and total mine count are inconsistent")

    pool = IDPool(start_from=len(unknowns) + 1)
    clauses = CardEnc.equals(
        lits=list(variables.values()),
        bound=remaining_mines,
        vpool=pool,
        encoding=EncType.seqcounter,
    ).clauses
    for index, clue in enumerate(observation.cells):
        if clue < 0:
            continue
        neighbours = _neighbours(index, observation.n_rows, observation.n_cols)
        neighbouring_flags = sum(observation.cells[neighbour] == FLAGGED for neighbour in neighbours)
        neighbouring_unknowns = [
            variables[neighbour]
            for neighbour in neighbours
            if observation.cells[neighbour] == UNKNOWN
        ]
        required_mines = clue - neighbouring_flags
        if not 0 <= required_mines <= len(neighbouring_unknowns):
            raise InconsistentObservation("A revealed clue contradicts the visible board")
        clauses.extend(CardEnc.equals(
            lits=neighbouring_unknowns,
            bound=required_mines,
            vpool=pool,
            encoding=EncType.seqcounter,
        ).clauses)

    solver = Glucose3(bootstrap_with=clauses)
    if not solver.solve():
        solver.delete()
        raise InconsistentObservation("No mine placement satisfies the visible board")
    return solver, variables


def _forced_orbits(observation: Observation, solver: Glucose3, variables: dict[int, int]):
    transforms = state_stabilizer(observation.cells, observation.n_rows, observation.n_cols)
    constrained = {
        neighbour
        for index, clue in enumerate(observation.cells)
        if clue >= 0
        for neighbour in _neighbours(index, observation.n_rows, observation.n_cols)
        if neighbour in variables
    }
    orbits = list(cell_orbits(constrained, transforms))
    unconstrained = tuple(sorted(set(variables) - constrained))
    if unconstrained:
        orbits.append(unconstrained)
    for orbit in sorted(orbits, key=lambda members: members[0]):
        variable = variables[orbit[0]]
        can_be_mine = solver.solve(assumptions=[variable])
        can_be_safe = solver.solve(assumptions=[-variable])
        if not can_be_mine:
            yield orbit, "reveal"
        elif not can_be_safe:
            yield orbit, "flag"


def deduce_all(observation: Observation) -> tuple[Action, ...]:
    solver, variables = _constraint_model(observation)
    try:
        actions = [
            Action(kind, divmod(index, observation.n_cols), "forced")
            for orbit, kind in _forced_orbits(observation, solver, variables)
            for index in orbit
        ]
        return tuple(sorted(actions, key=lambda action: action.cell))
    finally:
        solver.delete()


def choose_action(observation: Observation) -> Action | None:
    solver, variables = _constraint_model(observation)
    try:
        for orbit, kind in _forced_orbits(observation, solver, variables):
            return Action(kind, divmod(orbit[0], observation.n_cols), "forced")
        if variables:
            return Action("reveal", divmod(min(variables), observation.n_cols), "guess")
        return None
    finally:
        solver.delete()
