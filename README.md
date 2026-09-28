# Minesweeper solver

A Minesweeper player that reveals cells by deduction and guesses only when it has to. At each move it reads the visible board, plays any forced move it can find, and otherwise reveals the covered cell least likely to be a mine. The repository holds the solver package, a PyQt5 GUI, a test suite and a measurement script with its results.

The design comes from my BSc final project at the University of Leeds, "Automatic Minesweeper Solver & AI" (report submitted March 2025, supervisor Dr Adrian Martin). The code here is the 2026 implementation of that design, with tests and measurements.

## How it plays

At each move the solver reads only the visible board: clues, covered cells and flags. It never looks at the hidden mines. It then finds every forced move, plays one of them, or guesses when there is none.

- Forced moves. The covered cells next to clues split into independent regions. Each region is solved on its own clues with SAT (PySAT, Glucose3, with cardinality constraints), or taken from the cache. The whole board with the mine count is added only when the count can bind.
- Which move to play. The solver plays the forced move justified by the simplest rule: a single-clue rule if one applies, else a pattern from the library, else any other forced move (labelled "SAT"). A single-clue or pattern move is played only if SAT also finds it forced. The stages are labels for the simplest rule that justifies each move, not the order of computation.
- Single-clue rules. A clue whose mines are all found makes its other covered neighbours safe. A clue with exactly as many covered neighbours as missing mines makes them all mines.
- Pattern library. Eleven patterns, each stored once and found in every rotation and reflection by graph isomorphism on which covered cells each clue sees. In eight of them one clue already decides its cells on its own, so in play the single-clue rule acts first. Every pattern move comes from the other three: 1-2-1, 1-2 and the 1-1 edge pattern.
- Cache. A solved region is stored by its canonical form: the smallest of its eight rotations and reflections, moved to the origin. A region seen again anywhere on the board, in any rotation or reflection, is not solved twice within a game, on any board shape. When the whole board is solved with the mine count, cells that a symmetry of the current position swaps are tested once.
- Guess. When nothing is forced, it reveals the covered cell least likely to be a mine, from exact counting of the mine layouts that fit the visible board.

## Results

`scripts/deductive_measure.py` plays the same boards with three players: the SAT core, the full solver, and the full solver with the SAT core's lowest-index guess. The first click and its neighbours are never mines. The "SAT core" (`scripts/baseline_sat_core.py`) is the first version of this code (September 2026): SAT on the whole board at every move, and the lowest-index covered cell as its guess. The "full solver" is the pipeline above. The third player shows how much of the difference comes from the guess rule.

| Board | Boards played | SAT core | Full solver | Won by the full solver only / by the SAT core only | McNemar exact p |
|---|---|---|---|---|---|
| 8x8, 10 mines | 2,000 | 86.9% | 89.1% | 62 / 18 | 8.1e-7 |
| 16x16, 40 mines | 600 | 81.3% | 84.5% | 28 / 9 | 0.0026 |
| 16x30, 99 mines | 200 | 33.5% | 46.0% | 33 / 8 | 1.1e-4 |

With the SAT core's guess in place of its own, the full solver wins exactly the same boards as the SAT core, so the gain comes from the guess rule. On 8x8 boards, 89% of moves are justified by a single-clue rule, 6% by a pattern and 3% only by SAT, and 2% are guesses. Full results per board size are in `results/deductive_<level>.json`.

## Running it

Python 3.10 or later. Install the requirements first:

```
pip install -r requirements.txt
```

GUI:

```
python Minesweeper_Solver.py
```

This opens a board-size choice (Beginner 8x8, Intermediate 16x16, Expert 16x30, Extreme). "Start AI" plays the solver. "Show AI Logs" opens a log window that shows each move and the stage that justified it. The first cell revealed in a game, by a person or by the solver, is never a mine or next to one. Closing the game closes its log window.

Tests:

```
python -m pytest
```

This runs 123 tests. The tests check the SAT core, regions, patterns and probabilities against a full enumeration of the mine layouts on small boards. They also check that moves turn with the board under its symmetries, the play pipeline and the GUI.

Measurements:

```
python scripts/deductive_measure.py beginner --games 2000 --seed 91
python scripts/deductive_measure.py intermediate --games 600 --seed 92
python scripts/deductive_measure.py expert --games 200 --seed 93
```

## Layout

- `minesweeper/`
  - `board.py`: board mechanics
  - `deductive.py`: SAT core, regions, cache
  - `patterns.py`: pattern matcher
  - `probability.py`: exact mine probabilities
  - `pipeline.py`: play order
  - `symmetry.py`: board symmetries and canonical forms
- `Graph_Pattern_Generation2.py`: the pattern library, drawn as small boards
- `Minesweeper_Solver.py`: the GUI
- `scripts/`: the measurement script and the SAT-only baseline
- `tests/`: the test suite
- `results/`: measurement output per board size
