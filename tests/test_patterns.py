"""The pattern stage: patterns matched on the visible board, every move checked by SAT."""

from collections import Counter
from itertools import combinations, permutations

import networkx as nx
import pytest

from Graph_Pattern_Generation2 import PATTERNS, pattern_from_drawing
from minesweeper.deductive import FLAGGED, UNKNOWN, InconsistentObservation, Observation
from minesweeper.patterns import (
    PatternMoves,
    Proposal,
    find_pattern_matches,
    incidence_graph,
    node_match,
    pattern_conclusions,
    pattern_graphs,
    pattern_moves,
    propose,
    visible_graph,
)
from minesweeper.symmetry import grid_transforms
from test_regions import neighbours, positions_from_play

U = UNKNOWN
# A deliberately unsound pattern: a lone 1 beside two covered cells proves nothing, so its
# claims are often wrong. It keeps the rejection path tested.
UNSOUND_PATTERN = {"unsound_edge_pattern": {"graph": pattern_from_drawing("1 M\n. S"), "priority": 99}}
# Every drawing in the library, variants included, as (pattern name, drawing number).
DRAWINGS = [(name, number) for name, pattern in sorted(PATTERNS.items())
            for number in range(len(pattern_graphs(pattern)))]


def brute_force_forced(observation):
    """Try every placement of the remaining mines; a cell is forced if all consistent ones agree."""
    n_rows, n_cols, cells = observation.n_rows, observation.n_cols, observation.cells
    unknown = [i for i, s in enumerate(cells) if s == UNKNOWN]
    clues = []
    for i, s in enumerate(cells):
        if s >= 0:
            around = neighbours(i, n_rows, n_cols)
            clues.append((s - sum(cells[n] == FLAGGED for n in around),
                          [n for n in around if cells[n] == UNKNOWN]))
    values = {i: set() for i in unknown}
    for layout in combinations(unknown, observation.total_mines - cells.count(FLAGGED)):
        mines = set(layout)
        if all(sum(n in mines for n in around) == need for need, around in clues):
            for i in unknown:
                values[i].add(i in mines)
    return {("reveal" if seen == {False} else "flag", divmod(i, n_cols))
            for i, seen in values.items() if len(seen) == 1}


def transformed(observation, transform):
    cells = [None] * len(observation.cells)
    for index, image in enumerate(transform.mapping):
        cells[image] = observation.cells[index]
    return Observation(observation.n_rows, observation.n_cols, observation.total_mines, tuple(cells))


def board_from_drawing(graph, size):
    """A square board holding a pattern's drawing one cell in from the edge. Every other
    cell is revealed and shows 8, which no pattern looks for."""
    cells = [8] * (size * size)
    for (row, column), data in graph.nodes(data=True):
        cells[(row + 1) * size + column + 1] = data["clue"] if data["role"] == "clue" else UNKNOWN
    return Observation(size, size, 0, tuple(cells))


def pair_pattern(role):
    """A one-clue pattern: a revealed 1 next to one cell with the given role."""
    graph = nx.Graph()
    graph.add_node("clue", role="clue", clue=1, covered=False)
    graph.add_node("cell", role=role, covered=True)
    graph.add_edge("clue", "cell")
    return {"pair": {"graph": graph, "priority": 1,
                     "candidate_extractor": lambda board: [set(edge) for edge in board.edges]}}


@pytest.mark.parametrize(("name", "number"), DRAWINGS)
def test_each_drawing_is_a_valid_inference(name, number):
    graph = pattern_graphs(PATTERNS[name])[number]
    roles = {n: d["role"] for n, d in graph.nodes(data=True) if d["role"] != "clue"}
    clues = [n for n, d in graph.nodes(data=True) if d["role"] == "clue"]
    assert pattern_conclusions(graph) == roles  # the drawn roles are exactly what the clues force
    assert any(role != "unknown" for role in roles.values())  # and they prove something
    assert all(any(m in clues for m in graph.neighbors(n)) for n in roles)  # every covered cell is seen
    assert nx.is_connected(incidence_graph(graph))  # candidates link clues through cells they both see
    king = {(u, v) for u in graph for v in graph if u < v and max(abs(u[0] - v[0]), abs(u[1] - v[1])) == 1}
    assert {tuple(sorted(edge)) for edge in graph.edges} == king  # neighbouring cells are joined


def test_the_library_has_the_drawings_chosen():
    # Main drawing plus variants; a lost variant would otherwise just drop out of the
    # per-drawing tests.
    assert {name: len(pattern_graphs(pattern)) for name, pattern in PATTERNS.items()} == {
        "1_1_shared_mine": 2, "1_2_1": 3, "edge_2": 1, "edge_pattern": 1, "2_3_2_wall": 1,
        "false_edge_5": 1, "1_2": 2, "2_2": 2, "corner_1": 1, "1_1_1": 1, "triangle_1_1_1": 1}


def test_no_drawing_repeats_another():
    # Two drawings whose clues see their cells the same way would be the same rule
    # matched twice, however differently they are drawn.
    same = lambda a, b: a["role"] == b["role"] and a.get("clue") == b.get("clue")
    graphs = [(name, number, incidence_graph(pattern_graphs(PATTERNS[name])[number])) for name, number in DRAWINGS]
    for (name_a, number_a, a), (name_b, number_b, b) in combinations(graphs, 2):
        assert not nx.is_isomorphic(a, b, node_match=same), (name_a, number_a, name_b, number_b)


def test_pattern_conclusions_on_known_drawings():
    # A lone 1 beside two covered cells proves nothing.
    assert pattern_conclusions(pattern_from_drawing("1 ?\n. ?")) == {(0, 1): "unknown", (1, 1): "unknown"}
    # A boxed-in 1-2-1: the mines are beside the 1s.
    assert pattern_conclusions(pattern_from_drawing("1 2 1\nM S M")) == {
        (1, 0): "mine", (1, 1): "safe", (1, 2): "mine"}
    # A 1-2-1 over one covered cell. The 2 sees only that cell, so no layout fits.
    assert pattern_conclusions(pattern_from_drawing("1 2 1\n. M .")) is None


def test_the_5_on_a_board_of_its_own():
    # false_edge_5 rarely appears in play, so its moves are checked by SAT here.
    board = Observation(2, 3, 5, (U, 5, U, U, U, U))
    assert set(pattern_moves(board).certified) == {
        Proposal("false_edge_5", "flag", cell) for cell in [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)]}
    assert pattern_moves(board).rejected == ()


@pytest.mark.parametrize(("name", "number"), DRAWINGS)
def test_each_drawing_is_found_in_every_orientation(name, number):
    graph = pattern_graphs(PATTERNS[name])[number]
    size = max(max(row, column) for row, column in graph.nodes) + 3
    board = board_from_drawing(graph, size)
    drawn = {(name, "flag" if d["role"] == "mine" else "reveal", (row + 1, column + 1))
             for (row, column), d in graph.nodes(data=True) if d["role"] in ("mine", "safe")}
    transforms = grid_transforms(size, size)
    assert len(transforms) == 8
    for transform in transforms:
        found = {(p.pattern, p.kind, p.cell) for p in propose(transformed(board, transform))}
        assert {(pattern, kind, transform(*cell)) for pattern, kind, cell in drawn} <= found


def test_certified_moves_are_forced_and_rejected_moves_are_not():
    patterns = {**PATTERNS, **UNSOUND_PATTERN}
    verdicts = Counter()
    for n_rows, n_cols, num_mines, games in [(4, 4, 3, 200), (5, 5, 4, 200), (5, 5, 5, 150)]:
        for observation in positions_from_play(n_rows, n_cols, num_mines, games, seed=n_rows * 10 + num_mines):
            result = pattern_moves(observation, patterns)
            if not (result.certified or result.rejected):
                continue
            forced = brute_force_forced(observation)
            assert all((p.kind, p.cell) in forced for p in result.certified)
            assert not any((p.kind, p.cell) in forced for p in result.rejected)
            assert all(p.pattern == "unsound_edge_pattern" for p in result.rejected)  # the library is never wrong
            verdicts.update(("certified", p.kind) for p in result.certified)
            verdicts.update(("rejected", p.kind) for p in result.rejected)
    for verdict in ("certified", "rejected"):  # every verdict really occurs for both kinds of move
        for kind in ("flag", "reveal"):
            assert verdicts[verdict, kind] > 300, (verdict, kind, verdicts)


def test_the_1_2_1_along_an_open_row():
    # Covered row above, 1-2-1 in the middle of a revealed row: the open variant
    # finds the safe cell above the 2 and the cells beyond the 1s.
    board = Observation(3, 7, 2, (U,) * 7 + (0, 1, 1, 2, 1, 1, 0) + (0,) * 7)
    moves = pattern_moves(board)
    assert {(p.kind, p.cell) for p in moves.certified if p.pattern == "1_2_1"} == {
        ("reveal", (0, 1)), ("flag", (0, 2)), ("reveal", (0, 3)), ("flag", (0, 4)), ("reveal", (0, 5))}
    assert moves.rejected == ()


def test_a_pattern_beside_a_flag():
    # The flag is counted in the clues: the first 2 has one mine left, so 2-2-1 beside
    # the flag reads as a boxed-in 1-2-1, and the flagged cell is not part of the match.
    board = Observation(2, 4, 3, (FLAGGED, U, U, U, 2, 2, 2, 1))
    moves = pattern_moves(board)
    assert {(p.kind, p.cell) for p in moves.certified if p.pattern == "1_2_1"} == {
        ("flag", (0, 1)), ("reveal", (0, 2)), ("flag", (0, 3))}
    assert moves.rejected == ()


def test_the_1_1_with_the_safe_cell_touching_the_mine():
    # The 1 at the right edge sees only (1, 2); the 1 above it sees (1, 2) and (0, 3),
    # which touch. The 1-1 variant is drawn with the two cells apart ("1 1 . / M . S"),
    # but its clues see the same cells, so it finds the safe cell here.
    board = Observation(3, 4, 4, (U, U, 2, U, 3, U, U, 1, U, U, 1, 1))
    moves = pattern_moves(board)
    assert Proposal("1_1_shared_mine", "reveal", (0, 3)) in moves.certified
    assert moves.rejected == ()


def test_matching_depends_only_on_what_each_clue_sees():
    # The 2 at (0, 1) sees only (0, 0) and (1, 2), which do not touch; the 2 at (1, 1)
    # sees those two and (2, 1). The 2-2 variant is drawn with its two mines touching
    # ("2 M / 2 M / . S"), but its clues see the same cells, so it applies here.
    board = Observation(3, 3, 2, (U, 2, 1, 1, 2, U, 0, U, 1))
    moves = pattern_moves(board)
    assert {(p.kind, p.cell) for p in moves.certified if p.pattern == "2_2"} == {
        ("flag", (0, 0)), ("flag", (1, 2)), ("reveal", (2, 1))}
    assert moves.rejected == ()


def placements_by_brute_force(observation, drawing):
    """Every map from board cells to the drawing's nodes under which each clue, with its
    mines still to find, sees exactly the covered cells its node sees, found by trying
    every assignment of board cells to the drawing's nodes."""
    n_rows, n_cols, cells = observation.n_rows, observation.n_cols, observation.cells
    clue_nodes = [n for n, d in drawing.nodes(data=True) if d["role"] == "clue"]
    cell_nodes = [n for n, d in drawing.nodes(data=True) if d["role"] != "clue"]
    sees = {c: {m for m in drawing.neighbors(c) if m in cell_nodes} for c in clue_nodes}
    open_cells = {i: {n for n in neighbours(i, n_rows, n_cols) if cells[n] == UNKNOWN} for i in range(len(cells))}
    left = {i: s - sum(cells[n] == FLAGGED for n in neighbours(i, n_rows, n_cols))
            for i, s in enumerate(cells) if s >= 0}
    wanted = {drawing.nodes[c]["clue"] for c in clue_nodes}
    found = set()
    for chosen in permutations([i for i in left if left[i] in wanted], len(clue_nodes)):
        if any(left[i] != drawing.nodes[c]["clue"] for c, i in zip(clue_nodes, chosen)):
            continue
        covered = set().union(*(open_cells[i] for i in chosen))
        if len(covered) != len(cell_nodes):
            continue
        for image in permutations(sorted(covered)):
            to_board = dict(zip(cell_nodes, image))
            if all({to_board[m] for m in sees[c]} == open_cells[i] for c, i in zip(clue_nodes, chosen)):
                mapping = {divmod(i, n_cols): c for c, i in zip(clue_nodes, chosen)}
                mapping.update((divmod(i, n_cols), m) for m, i in to_board.items())
                found.add(frozenset(mapping.items()))
    return found


def test_placements_equal_a_brute_force_search_for_what_each_clue_sees():
    placements, drawn_differently = Counter(), Counter()
    for n_rows, n_cols, num_mines, seed in [(5, 5, 4, 11), (6, 6, 7, 12), (5, 6, 6, 13)]:
        for observation in positions_from_play(n_rows, n_cols, num_mines, games=25, seed=seed):
            graph = visible_graph(observation)
            for name, number in DRAWINGS:
                drawing = pattern_graphs(PATTERNS[name])[number]
                expected = placements_by_brute_force(observation, drawing)
                found = {frozenset(mapping.items()) for used, mapping in find_pattern_matches(graph, {"graph": drawing})}
                assert found == expected, (name, number, observation)
                placements[name] += len(expected)
                drawn_differently[name] += sum(map(touch_differently(graph, drawing), expected))
    assert sum(placements.values()) > 1000
    assert sum(1 for count in drawn_differently.values() if count) >= 6  # at least six were found in arrangements not drawn


def touch_differently(graph, drawing):
    """Whether a placement's cells touch one another differently from the drawing."""
    return lambda mapping: any(graph.has_edge(a, b) != drawing.has_edge(node_a, node_b)
                               for (a, node_a), (b, node_b) in combinations(mapping, 2))


@pytest.mark.parametrize(("drawing", "board"), [
    # The 2-3-2 bent into an L around its covered cells.
    (PATTERNS["2_3_2_wall"]["graph"], Observation(5, 5, 3, (U, 1, 1, 1, U, 0, 2, U, 2, 0, 0, 3, U, 3, 0,
                                                            0, U, U, 2, 0, 0, 1, 1, 1, U))),
    # The 1-2-1 open at both ends, bent across two rows.
    (PATTERNS["1_2_1"]["variants"][1], Observation(4, 6, 3, (0, U, 0, 0, 0, 0, 0, 1, 1, U, 0, U,
                                                             1, 2, U, 2, 1, 0, U, 2, 2, U, 1, 0))),
    # Three clues in a chain, the outer two seeing no cell in common, with the middle clue
    # first in board order under one symmetry and last under another: the groups of
    # clues must grow from any clue already in them.
    (pattern_from_drawing("1 . 1 . 1\n? ? ? ? ?"), Observation(4, 6, 3, (1, U, U, 1, 0, U, 1, U, U, 1, 0, 0,
                                                                          0, 1, 1, 2, 1, U, U, 0, 0, U, U, 1))),
])
def test_placements_in_arrangements_not_drawn(drawing, board):
    for transform in grid_transforms(board.n_rows, board.n_cols):
        moved = transformed(board, transform)
        expected = placements_by_brute_force(moved, drawing)
        graph = visible_graph(moved)
        assert {frozenset(mapping.items()) for _, mapping in find_pattern_matches(graph, {"graph": drawing})} == expected
        assert any(map(touch_differently(graph, drawing), expected))


def test_the_1_2_1_against_a_wall():
    # The mines are beside the 1s and the cell beside the 2 is safe. Each 1 with the 2
    # is also a 1-2 with the 1 at a closed end, which finds the same two mines.
    wall = Observation(2, 3, 2, (U, U, U, 1, 2, 1))
    moves = pattern_moves(wall)
    assert set(moves.certified) == {
        Proposal("1_2_1", "flag", (0, 0)), Proposal("1_2_1", "reveal", (0, 1)), Proposal("1_2_1", "flag", (0, 2)),
        Proposal("1_2", "flag", (0, 0)), Proposal("1_2", "flag", (0, 2))}
    assert [p.pattern for p in moves.certified] == ["1_2_1"] * 3 + ["1_2"] * 2  # in priority order
    assert moves.rejected == ()


def test_each_proposal_is_checked_on_its_own():
    # The unsound pattern's graph has a symmetry that swaps its mine and safe cells, so
    # it claims both answers for each of its two cells; the board decides which is right.
    board = Observation(2, 3, 1, (U, 1, 0, 1, U, U))
    assert pattern_moves(board, UNSOUND_PATTERN) == PatternMoves(
        certified=(Proposal("unsound_edge_pattern", "flag", (0, 0)),
                   Proposal("unsound_edge_pattern", "reveal", (1, 1))),
        rejected=(Proposal("unsound_edge_pattern", "flag", (1, 1)),
                  Proposal("unsound_edge_pattern", "reveal", (0, 0))),
    )


def test_roles_match_only_the_cells_they_can_stand_for():
    # Flags are counted in the clues: the 1 next to the flag has no mine left to find,
    # and the flagged cell itself matches no covered role. Revealed cells match only clues.
    board = Observation(1, 4, 2, (FLAGGED, 1, 1, U))
    assert propose(board, pair_pattern("mine")) == (Proposal("pair", "flag", (0, 3)),)
    assert propose(board, pair_pattern("safe")) == (Proposal("pair", "reveal", (0, 3)),)
    assert propose(board, pair_pattern("unknown")) == ()
    assert pattern_moves(board, pair_pattern("mine")) == PatternMoves((Proposal("pair", "flag", (0, 3)),), ())
    assert pattern_moves(board, pair_pattern("safe")) == PatternMoves((), (Proposal("pair", "reveal", (0, 3)),))


def test_a_placement_must_hold_every_covered_neighbour_of_its_clues():
    # The 1-2-1 cells are all present, but the 2 also sees a covered cell below it,
    # so the pattern's conclusion would not follow and it must not match.
    board = Observation(3, 3, 2, (U, U, U, 1, 2, 1, 8, U, 8))
    assert not any(p.pattern == "1_2_1" for p in propose(board))
    # The same with an extractor that hands over only the drawn cells: the placement
    # is isomorphic to the pattern, and the neighbourhood check alone rejects it.
    drawn_cells = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}
    only_drawn = {"1_2_1": {**PATTERNS["1_2_1"], "candidate_extractor": lambda graph: [drawn_cells]}}
    assert propose(board, only_drawn) == ()
    closed_below = Observation(3, 3, 2, (U, U, U, 1, 2, 1, 8, 8, 8))
    assert set(propose(closed_below, only_drawn)) == {
        Proposal("1_2_1", "flag", (0, 0)), Proposal("1_2_1", "reveal", (0, 1)), Proposal("1_2_1", "flag", (0, 2))}


def test_the_board_graph_shows_what_the_player_sees():
    graph = visible_graph(Observation(1, 3, 1, (U, FLAGGED, 2)))
    assert dict(graph.nodes[(0, 0)]) == {"clue": -1, "mines_left": -1, "covered": True, "flagged": False}
    assert dict(graph.nodes[(0, 1)]) == {"clue": -1, "mines_left": -1, "covered": True, "flagged": True}
    assert dict(graph.nodes[(0, 2)]) == {"clue": 2, "mines_left": 1, "covered": False, "flagged": False}
    assert set(graph.edges) == {((0, 0), (0, 1)), ((0, 1), (0, 2))}
    assert (graph.graph["n_rows"], graph.graph["n_cols"]) == (1, 3)


def test_a_clue_node_counts_the_mines_still_to_find():
    two_beside_a_flag = {"clue": 2, "mines_left": 1, "covered": False, "flagged": False}
    assert node_match(two_beside_a_flag, {"role": "clue", "clue": 1})
    assert not node_match(two_beside_a_flag, {"role": "clue", "clue": 2})
    flagged = {"clue": -1, "mines_left": -1, "covered": True, "flagged": True}
    assert not any(node_match(flagged, {"role": role}) for role in ("mine", "safe", "unknown"))


def test_each_placement_is_found_once():
    for observation in positions_from_play(9, 9, 10, games=3, seed=4):
        graph = visible_graph(observation)
        for pattern in PATTERNS.values():
            placements = [tuple(sorted(mapping.items())) for _, mapping in find_pattern_matches(graph, pattern)]
            assert len(placements) == len(set(placements))


def test_an_inconsistent_board_is_rejected_even_without_matches():
    impossible = Observation(1, 3, 1, (U, 3, U))
    assert propose(impossible) == ()
    with pytest.raises(InconsistentObservation):
        pattern_moves(impossible)


def test_patterns_follow_the_board_under_rotation_and_reflection():
    # On positions from play, where several patterns overlap, the proposals move with the board.
    transforms = grid_transforms(5, 5)
    seen = set()
    for observation in positions_from_play(5, 5, 4, games=40, seed=3):
        found = {(p.pattern, p.kind, p.cell) for p in propose(observation)}
        for transform in transforms:
            expected = {(name, kind, transform(*cell)) for name, kind, cell in found}
            assert {(p.pattern, p.kind, p.cell) for p in propose(transformed(observation, transform))} == expected
        seen |= {(name, kind) for name, kind, _ in found}
    assert len({name for name, _ in seen}) >= 6 and {kind for _, kind in seen} == {"flag", "reveal"}
