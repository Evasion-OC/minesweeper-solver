"""The board graph and the pattern matcher see only what a player sees."""

import copy
import random

from Graph_Pattern_Generation2 import PATTERNS
from minesweeper.board import (
    bfs_expand,
    build_adjacency,
    compute_clues,
    create_board,
    generate_random_board,
)
from minesweeper.deductive import Observation
from minesweeper.patterns import find_pattern_matches, pattern_graphs, visible_graph


class VisibleOnlyCell(dict):
    """A board cell that fails if the hidden mine or a covered cell's clue is read."""

    def _check(self, key):
        if key == "isMine" or (key == "clue" and dict.__getitem__(self, "covered")):
            raise AssertionError(f"hidden state read: {key}")

    def __getitem__(self, key):
        self._check(key)
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        self._check(key)
        return dict.get(self, key, default)


def reveal(board, adj, r, c):
    board[r][c]["covered"] = False
    bfs_expand(board, r, c, adj)


def played_board(seed, n_rows=8, n_cols=8, num_mines=10, extra_reveals=6):
    """A random board after a safe first click and a few more safe reveals."""
    rng = random.Random(seed)
    random.seed(seed)  # generate_random_board draws from the global generator
    first = (rng.randrange(n_rows), rng.randrange(n_cols))
    grid = generate_random_board(n_rows, n_cols, num_mines, first_move=first)
    compute_clues(grid)
    board = create_board(grid)
    adj = build_adjacency(board)
    reveal(board, adj, *first)
    safe = [(r, c) for r in range(n_rows) for c in range(n_cols)
            if board[r][c]["covered"] and not board[r][c]["isMine"]]
    for r, c in rng.sample(safe, min(extra_reveals, len(safe))):
        if board[r][c]["covered"]:
            reveal(board, adj, r, c)
    return board, adj


def all_matches(board, num_mines=10):
    graph = visible_graph(Observation.from_board(board, num_mines))
    return {
        name: sorted(sorted(mapping.items()) for _, mapping in find_pattern_matches(graph, info))
        for name, info in PATTERNS.items()
    }


def test_pattern_nodes_carry_roles_not_hidden_state():
    for name, info in PATTERNS.items():
        for node, data in (item for drawing in pattern_graphs(info) for item in drawing.nodes(data=True)):
            assert "isMine" not in data, (name, node)
            assert data["role"] in ("clue", "mine", "safe", "unknown"), (name, node)
            assert data["covered"] == (data["role"] != "clue"), (name, node)
            if data["role"] == "clue":
                assert 0 <= data["clue"] <= 8, (name, node)


def test_graph_building_and_matching_never_read_hidden_state():
    concluding = {name for name, info in PATTERNS.items()
                  if any(data["role"] in ("mine", "safe")
                         for drawing in pattern_graphs(info) for _, data in drawing.nodes(data=True))}
    total = 0
    for seed in range(40):
        board, adj = played_board(seed)
        guarded = [[VisibleOnlyCell(cell) for cell in row] for row in board]
        matches = all_matches(guarded)
        total += sum(len(matches[name]) for name in concluding)
    assert total > 0  # patterns that draw conclusions really matched on these boards


def test_matches_depend_only_on_visible_state():
    for seed in range(40):
        board, adj = played_board(seed)
        other = copy.deepcopy(board)
        rng = random.Random(1000 + seed)
        for row in other:
            for cell in row:
                if cell["covered"]:  # rewrite what the player cannot see
                    cell["isMine"] = not cell["isMine"]
                    cell["clue"] = rng.randrange(-1, 9)
        assert all_matches(board) == all_matches(other)
