"""The deductive solver against its SAT core alone, on the same boards. Writes
results/deductive_<level>.json.

    python scripts/deductive_measure.py beginner --games 2000 --seed 91
    python scripts/deductive_measure.py intermediate --games 600 --seed 92
    python scripts/deductive_measure.py expert --games 200 --seed 93

Levels are the repository's (minesweeper.board.DIFFICULTY_PARAMS). Every player gets the
same boards: the first cell is chosen at random and no mine lies on it or next to it; the
first N games of a seed are the same boards whatever the total.

Players:
- SAT core: scripts/baseline_sat_core.py, the SAT baseline. It solves the whole
  board by SAT at every move, plays a forced move if there is one, and otherwise reveals
  the lowest-index covered cell.
- full solver: minesweeper.pipeline.PlaySolver with a fresh cache of solved regions for
  every game. When nothing is forced it reveals the covered cell least likely to be a mine.
- full solver, lowest-index guess: the full solver's deduction with the SAT core's guess.
  Both deductions find every forced move, so the two players reach the same position
  whenever nothing is forced; this player shows how much of the difference in wins comes
  from the guess rule.

The full solver plays each game twice: once with counters around its region solve and
whole-board model (moves by stage, region lookups, symmetry, guess calibration), and once
without them for the time per move, so the counting is not timed. Both passes must play
the same game.

Recorded: wins, with the boards only one of the SAT core and the full solver won and an
exact McNemar test on them; moves and guesses per game; time per move; moves by stage;
each region lookup classed against the regions seen earlier in the same game (same place,
moved, or rotated or reflected); regions with a symmetry of their own, those whose
symmetry moves a covered cell, and the SAT cell tests the symmetry saves; positions whose whole
board has a symmetry of its own; whole-board models built; and the calibration of the
guesses: predicted mine probability against the share that were mines, overall and in
bins of predicted probability.
"""

import argparse
import json
import random
import statistics
import sys
import time
import types
from collections import Counter
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import minesweeper.deductive as core  # noqa: E402
from minesweeper.board import DIFFICULTY_PARAMS  # noqa: E402
from minesweeper.pipeline import STAGES, PlaySolver  # noqa: E402
from minesweeper.probability import TooManyLayouts, mine_probabilities  # noqa: E402
from minesweeper.symmetry import canonical_form, state_stabilizer  # noqa: E402

LEVELS = ("beginner", "intermediate", "expert")
BASELINE = ROOT / "scripts" / "baseline_sat_core.py"
LOOKUP_KINDS = ("same region, same place", "same picture, moved", "same picture, rotated or reflected", "new")


def sat_core_only():
    """scripts/baseline_sat_core.py, loaded as its own module."""
    module = types.ModuleType("deductive_sat_core_only")
    sys.modules[module.__name__] = module
    exec(compile(BASELINE.read_text(), str(BASELINE), "exec"), module.__dict__)
    return module


def play(rng, n_rows, n_cols, num_mines, choose):
    """One game on a board drawn from rng; returns (won, moves, guesses)."""
    first = rng.randrange(n_rows * n_cols)
    banned = set(core._neighbours(first, n_rows, n_cols)) | {first}
    mines = set(rng.sample([i for i in range(n_rows * n_cols) if i not in banned], num_mines))
    clue = [sum(n in mines for n in core._neighbours(i, n_rows, n_cols)) for i in range(n_rows * n_cols)]
    visible = [core.UNKNOWN] * (n_rows * n_cols)

    def reveal(index):
        stack = [index]
        while stack:
            cell = stack.pop()
            if visible[cell] != core.UNKNOWN:
                continue
            visible[cell] = clue[cell]
            if clue[cell] == 0:
                stack.extend(core._neighbours(cell, n_rows, n_cols))

    reveal(first)
    moves = guesses = 0
    while True:
        if all(visible[i] != core.UNKNOWN for i in range(n_rows * n_cols) if i not in mines):
            return True, moves, guesses
        action = choose(tuple(visible), mines)
        moves += 1
        guesses += action.reason == "guess"
        index = action.cell[0] * n_cols + action.cell[1]
        if action.kind == "flag":
            visible[index] = core.FLAGGED
        elif index in mines:
            return False, moves, guesses
        else:
            reveal(index)


def milliseconds(seconds):
    ordered = sorted(seconds)
    return {"mean": round(1000 * statistics.mean(ordered), 3),
            "median": round(1000 * statistics.median(ordered), 3),
            "p95": round(1000 * ordered[int(0.95 * len(ordered))], 3)}


def calibration_summary(pairs):
    """pairs are (predicted mine probability, was a mine), one per guess."""
    def summary(group):
        return {"guesses": len(group),
                "mean predicted mine probability": sum(p for p, _ in group) / len(group),
                "share that were mines": sum(m for _, m in group) / len(group)}

    bins = {}
    for predicted, was_mine in pairs:
        low = min(int(predicted * 10), 9) / 10
        bins.setdefault(f"{low:.1f}-{low + 0.1:.1f}", []).append((predicted, was_mine))
    return {**(summary(pairs) if pairs else {"guesses": 0}),
            "by predicted probability": {name: summary(group) for name, group in sorted(bins.items())}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("level", choices=LEVELS)
    p.add_argument("--games", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", default=None, help="default: results/deductive_<level>.json")
    args = p.parse_args()
    if args.games < 1:
        p.error("--games must be at least 1")
    n_rows, n_cols, num_mines = DIFFICULTY_PARAMS[args.level]
    baseline = sat_core_only()

    # Counters around the full solver's region solve and whole-board model, switched on
    # for the counting pass only. Lookups are classed against the pictures seen earlier
    # in the same game, since the cache is per game.
    lookups, seen, counts = Counter(), {"place": set(), "moved": set(), "form": set()}, Counter()
    solve_region, constraint_model = core._solve_region, core._constraint_model

    def counted_solve_region(observation, region, cache):
        picture = core._region_picture(observation, region)
        place = tuple(sorted(picture))
        top, left = min(r for r, _, _ in picture), min(c for _, c, _ in picture)
        moved = tuple(sorted((r - top, c - left, label) for r, c, label in picture))
        form, placements = canonical_form(picture)
        kind = ("same region, same place" if place in seen["place"]
                else "same picture, moved" if moved in seen["moved"]
                else "same picture, rotated or reflected" if form in seen["form"] else "new")
        lookups[kind] += 1
        if kind == "new":
            orbits = {frozenset(placement[divmod(index, observation.n_cols)] for placement in placements)
                      for index in region.cells}
            counts["regions solved"] += 1
            counts["solved regions with a symmetry of their own"] += len(placements) > 1
            counts["solved regions whose symmetry moves a covered cell"] += len(orbits) < len(region.cells)
            counts["covered cells in solved regions"] += len(region.cells)
            counts["cell tests saved by region symmetry"] += len(region.cells) - len(orbits)
        seen["place"].add(place)
        seen["moved"].add(moved)
        seen["form"].add(form)
        return solve_region(observation, region, cache)

    def counted_constraint_model(observation):
        counts["whole-board models built"] += 1
        return constraint_model(observation)

    def counting(on):
        core._solve_region, core._constraint_model = ((counted_solve_region, counted_constraint_model) if on
                                                      else (solve_region, constraint_model))

    baseline_times = []

    def baseline_choice(cells, mines):
        observation = baseline.Observation(n_rows, n_cols, num_mines, cells)
        start = time.perf_counter()
        action = baseline.choose_action(observation)
        baseline_times.append(time.perf_counter() - start)
        return action

    solver, calibration = PlaySolver(), []

    def counted_choice(cells, mines):
        observation = core.Observation(n_rows, n_cols, num_mines, cells)
        counts["positions"] += 1
        counts["positions whose board has a symmetry of its own"] += len(state_stabilizer(cells, n_rows, n_cols)) > 1
        step = solver.step(observation)
        if step.stage == "guess":
            index = step.action.cell[0] * n_cols + step.action.cell[1]
            try:
                calibration.append((float(mine_probabilities(observation)[index]), index in mines))
            except TooManyLayouts:
                counts["guesses with a region too large to count"] += 1
        return step.action

    timed_solver, solver_times = PlaySolver(), []

    def timed_choice(cells, mines):
        observation = core.Observation(n_rows, n_cols, num_mines, cells)
        start = time.perf_counter()
        step = timed_solver.step(observation)
        solver_times.append(time.perf_counter() - start)
        return step.action

    check_cache = core.RegionCache()

    def lowest_index_guess_choice(cells, mines):
        forced = core.deduce_all(core.Observation(n_rows, n_cols, num_mines, cells), check_cache)
        if forced:
            return forced[0]
        return core.Action("reveal", divmod(cells.index(core.UNKNOWN), n_cols), "guess")

    rngs = [random.Random(args.seed) for _ in range(4)]
    games = []
    for _ in range(args.games):
        baseline_game = play(rngs[0], n_rows, n_cols, num_mines, baseline_choice)
        counting(True)
        solver.new_game()
        for pictures in seen.values():
            pictures.clear()
        solver_game = play(rngs[1], n_rows, n_cols, num_mines, counted_choice)
        counting(False)
        timed_solver.new_game()
        if play(rngs[2], n_rows, n_cols, num_mines, timed_choice) != solver_game:
            raise RuntimeError("the timing pass played a different game from the counting pass")
        check_cache = core.RegionCache()
        check_game = play(rngs[3], n_rows, n_cols, num_mines, lowest_index_guess_choice)
        games.append((baseline_game, solver_game, check_game))

    def share(column, field):
        return sum(game[column][field] for game in games) / args.games

    only_solver = sum(1 for b, s, _ in games if s[0] and not b[0])
    only_baseline = sum(1 for b, s, _ in games if b[0] and not s[0])
    discordant = only_solver + only_baseline
    p_value = (min(1.0, 2 * sum(comb(discordant, i) for i in range(min(only_solver, only_baseline) + 1)) / 2 ** discordant)
               if discordant else 1.0)
    result = {
        "board": f"{n_rows}x{n_cols}/{num_mines}", "level": args.level, "games": args.games, "seed": args.seed,
        "players": {
            "SAT core": "scripts/baseline_sat_core.py; guesses the lowest-index covered cell",
            "full solver": "minesweeper.pipeline.PlaySolver; guesses the covered cell least likely to be a mine",
            "full solver, lowest-index guess": "the full solver's deduction with the SAT core's guess",
        },
        "win rate": {"SAT core": share(0, 0), "full solver": share(1, 0), "full solver, lowest-index guess": share(2, 0)},
        "boards won by only one of the SAT core and the full solver": {
            "full solver": only_solver, "SAT core": only_baseline, "McNemar exact p": p_value},
        "boards where the full solver with the lowest-index guess and the SAT core differ in result":
            sum(1 for b, _, c in games if b[0] != c[0]),
        "moves per game": {"SAT core": share(0, 1), "full solver": share(1, 1)},
        "guesses per game": {"SAT core": share(0, 2), "full solver": share(1, 2)},
        "time per move, ms": {"SAT core": milliseconds(baseline_times), "full solver": milliseconds(solver_times)},
        "full solver": {
            "moves by stage": {stage: solver.stage_counts[stage] for stage in STAGES},
            "positions": counts["positions"],
            "positions whose board has a symmetry of its own": counts["positions whose board has a symmetry of its own"],
            "whole-board models built": counts["whole-board models built"],
            "region lookups within a game": {kind: lookups[kind] for kind in LOOKUP_KINDS},
            **{key: counts[key] for key in ("regions solved", "solved regions with a symmetry of their own",
                                            "solved regions whose symmetry moves a covered cell",
                                            "covered cells in solved regions", "cell tests saved by region symmetry")},
            "guess calibration": {**calibration_summary(calibration),
                                  "guesses with a region too large to count":
                                      counts["guesses with a region too large to count"]},
        },
    }
    out = Path(args.out) if args.out else ROOT / "results" / f"deductive_{args.level}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
