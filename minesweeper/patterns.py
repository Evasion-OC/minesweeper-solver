"""The pattern library, matched on the visible board, with every move checked by SAT.

A pattern (Graph_Pattern_Generation2) is a small graph of revealed clue cells and
covered cells, built from a drawing so that neighbouring cells are always joined.
A clue node gives the number of mines still to find around that cell, and each
covered node has a role: 'mine' or 'safe' is what the pattern concludes about it.

A placement counts only when the pattern holds every covered neighbour of its
clues, so the clues constrain nothing outside the pattern. The conclusions then
follow from the clues alone, and pattern_conclusions checks them by trying every
layout of the pattern's covered cells. They depend only on which covered cells
each clue sees, so matching compares incidence graphs, in which each clue is
joined to the covered cells it sees and nothing else is joined. Matching is by
graph isomorphism, which ignores orientation, so a pattern stored once is found
in every rotation and reflection, and in every other arrangement where its clues
see the same cells, whether or not the cells touch one another as drawn. A
pattern may have variants, further drawings of the same rule; each is matched on
its own. Every proposed move is still checked against the whole board, clues and
mine count together, before it is certified.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Literal

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

from Graph_Pattern_Generation2 import PATTERNS
from minesweeper.deductive import FLAGGED, Coordinate, Observation, _constraint_model, _neighbours


def visible_graph(observation: Observation) -> nx.Graph:
    """The board as the player sees it: a node per cell, an edge between neighbours.

    A revealed cell carries its clue and the mines still to find around it, which
    is the clue less its flagged neighbours.
    """
    n_rows, n_cols, cells = observation.n_rows, observation.n_cols, observation.cells
    graph = nx.Graph(n_rows=n_rows, n_cols=n_cols)
    for index, state in enumerate(cells):
        around = _neighbours(index, n_rows, n_cols)
        graph.add_node(
            divmod(index, n_cols),
            clue=state if state >= 0 else -1,
            mines_left=state - sum(cells[n] == FLAGGED for n in around) if state >= 0 else -1,
            covered=state < 0,
            flagged=state == FLAGGED,
        )
        for neighbour in around:
            if neighbour > index:
                graph.add_edge(divmod(index, n_cols), divmod(neighbour, n_cols))
    return graph


def node_match(board_attrs, pattern_attrs) -> bool:
    """Match a board cell to a pattern node using visible state only.

    A 'clue' node matches a revealed cell with that many mines still to find.
    'mine', 'safe' and 'unknown' nodes match covered cells that are not flagged;
    flags are already counted in the clues. Whether the pattern's conclusion
    actually holds is not decided here.
    """
    if pattern_attrs['role'] == 'clue':
        return not board_attrs['covered'] and board_attrs['mines_left'] == pattern_attrs['clue']
    return board_attrs['covered'] and not board_attrs['flagged']


def _open_cells(graph: nx.Graph, cells) -> set:
    """The covered, unflagged neighbours of the given cells."""
    return {n for cell in cells for n in graph.neighbors(cell)
            if graph.nodes[n]['covered'] and not graph.nodes[n]['flagged']}


@lru_cache(maxsize=None)
def incidence_graph(pattern_graph: nx.Graph) -> nx.Graph:
    """The pattern with each clue joined to the covered cells it sees, and nothing else joined."""
    incidence = nx.Graph()
    incidence.add_nodes_from(pattern_graph.nodes(data=True))
    incidence.add_edges_from((u, v) for u, v in pattern_graph.edges
                             if (pattern_graph.nodes[u]['role'] == 'clue') != (pattern_graph.nodes[v]['role'] == 'clue'))
    return incidence


def _placed_incidence(graph: nx.Graph, cells) -> nx.Graph:
    """Board cells with each revealed cell joined to the covered cells among them it sees."""
    placed = nx.Graph()
    placed.add_nodes_from((cell, graph.nodes[cell]) for cell in cells)
    placed.add_edges_from((u, v) for u, v in graph.subgraph(cells).edges
                          if graph.nodes[u]['covered'] != graph.nodes[v]['covered'])
    return placed


def clue_group_candidates(graph: nx.Graph, pattern_graph: nx.Graph):
    """Candidate cell sets for a pattern: a group of revealed cells with the pattern's
    clue values, linked through covered cells they see in common, together with every
    covered neighbour of the group. A pattern's clues are linked the same way."""
    wanted = sorted(d['clue'] for _, d in pattern_graph.nodes(data=True) if d['role'] == 'clue')
    sees = {n: _open_cells(graph, [n]) for n, d in graph.nodes(data=True)
            if not d['covered'] and d['mines_left'] in wanted}
    sees = {clue: cells for clue, cells in sees.items() if cells}
    seen_by = {}
    for clue, cells in sees.items():
        for cell in cells:
            seen_by.setdefault(cell, set()).add(clue)
    linked = {clue: {other for cell in cells for other in seen_by[cell]} - {clue} for clue, cells in sees.items()}
    groups = {frozenset([clue]) for clue in sees}
    for _ in range(len(wanted) - 1):
        groups = {group | {other} for group in groups for clue in group
                  for other in linked[clue] if other not in group}
    for group in groups:
        if sorted(graph.nodes[clue]['mines_left'] for clue in group) == wanted:
            yield group | set().union(*(sees[clue] for clue in group))


def pattern_graphs(pattern: dict) -> tuple[nx.Graph, ...]:
    """A pattern's drawings: the main one, then its variants."""
    return (pattern['graph'], *pattern.get('variants', ()))


def find_pattern_matches(graph: nx.Graph, pattern: dict):
    """Every placement of the pattern on the board, as (drawing, map from board cell to drawing node).

    Each drawing of the pattern is matched on its own, and each distinct candidate
    cell set is tested once per drawing, comparing what each clue sees (the
    incidence graphs). A placement is kept only when it holds every covered
    neighbour of the cells it maps to clues. Every isomorphism is kept, so a drawing
    whose clues see its cells symmetrically gives one map per symmetry.
    """
    extractor = pattern.get('candidate_extractor')
    for drawing in pattern_graphs(pattern):
        wanted = incidence_graph(drawing)
        candidates = extractor(graph) if extractor else clue_group_candidates(graph, drawing)
        seen = set()
        for candidate in candidates:
            cells = frozenset(candidate)
            if cells in seen:
                continue
            seen.add(cells)
            if len(cells) != len(wanted):
                continue
            placed = _placed_incidence(graph, cells)
            if placed.number_of_edges() != wanted.number_of_edges():
                continue  # cannot be isomorphic; skip the full test
            matcher = GraphMatcher(placed, wanted, node_match=node_match)
            for mapping in matcher.isomorphisms_iter():
                clues = [cell for cell, node in mapping.items() if drawing.nodes[node]['role'] == 'clue']
                if _open_cells(graph, clues) <= cells:
                    yield drawing, mapping


def pattern_conclusions(pattern_graph: nx.Graph):
    """What a pattern's clues force on its covered cells, trying every layout of them.

    Returns a map from covered node to 'mine', 'safe' or 'unknown', or None when no
    layout satisfies the clues. Because a placement must hold every covered
    neighbour of its clues, these are exactly the moves the pattern proves.
    """
    covered = sorted(n for n, d in pattern_graph.nodes(data=True) if d['role'] != 'clue')
    position = {node: i for i, node in enumerate(covered)}
    clues = [(d['clue'], [position[m] for m in pattern_graph.neighbors(n) if m in position])
             for n, d in pattern_graph.nodes(data=True) if d['role'] == 'clue']
    layouts = [bits for bits in product((0, 1), repeat=len(covered))
               if all(sum(bits[i] for i in around) == need for need, around in clues)]
    if not layouts:
        return None
    return {node: 'mine' if all(bits[i] for bits in layouts) else
                  'safe' if not any(bits[i] for bits in layouts) else 'unknown'
            for node, i in position.items()}


@dataclass(frozen=True)
class Proposal:
    pattern: str
    kind: Literal["reveal", "flag"]
    cell: Coordinate


def propose(observation: Observation, patterns: dict = PATTERNS) -> tuple[Proposal, ...]:
    """The moves the patterns claim, in priority order, each pattern's claims listed once."""
    graph = visible_graph(observation)
    proposals = {}
    for name, pattern in sorted(patterns.items(), key=lambda item: item[1].get('priority', 100)):
        for drawing, mapping in find_pattern_matches(graph, pattern):
            for cell, node in mapping.items():
                role = drawing.nodes[node]['role']
                if role == 'safe':
                    proposals.setdefault(Proposal(name, "reveal", cell), None)
                elif role == 'mine':
                    proposals.setdefault(Proposal(name, "flag", cell), None)
    return tuple(proposals)


@dataclass(frozen=True)
class PatternMoves:
    certified: tuple[Proposal, ...]
    rejected: tuple[Proposal, ...]


def pattern_moves(observation: Observation, patterns: dict = PATTERNS) -> PatternMoves:
    """Split the patterns' proposals into moves proved forced and moves that are not.

    A move is forced when its opposite contradicts the board: revealing a cell is
    certified when the cell cannot be a mine, flagging it when it cannot be safe.
    A board no mine layout fits raises InconsistentObservation, whether or not a
    pattern matched.
    """
    proposals = propose(observation, patterns)
    solver, variables = _constraint_model(observation)  # also rejects inconsistent boards
    try:
        verdicts = {}
        certified, rejected = [], []
        for proposal in proposals:
            move = (proposal.kind, proposal.cell)
            if move not in verdicts:
                row, column = proposal.cell
                variable = variables[row * observation.n_cols + column]
                opposite = variable if proposal.kind == "reveal" else -variable
                verdicts[move] = not solver.solve(assumptions=[opposite])
            (certified if verdicts[move] else rejected).append(proposal)
    finally:
        solver.delete()
    return PatternMoves(tuple(certified), tuple(rejected))
