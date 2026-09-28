import networkx as nx

# Each pattern is drawn as text, one token per cell:
#   a digit  a revealed cell, with that many mines still to find around it
#            (its clue less the flags beside it)
#   M        a covered cell the pattern proves is a mine
#   S        a covered cell the pattern proves is safe
#   ?        a covered cell the pattern leaves undecided
#   .        not part of the pattern: a revealed cell, a flag, or outside the board
# How patterns are matched is described in minesweeper/patterns.py.
#
# A pattern can have variants: further drawings of the same rule, for instance with
# the covered row open at one or both ends. Each is matched on its own, and its moves
# are reported under the pattern's name.


def pattern_from_drawing(drawing):
    G = nx.Graph()
    for r, line in enumerate(drawing.strip().splitlines()):
        for c, token in enumerate(line.split()):
            if token.isdigit():
                G.add_node((r, c), role='clue', clue=int(token), covered=False)
            elif token in ('M', 'S', '?'):
                role = {'M': 'mine', 'S': 'safe', '?': 'unknown'}[token]
                G.add_node((r, c), role=role, covered=True)
            elif token != '.':
                raise ValueError(f"unknown token {token!r} in pattern drawing")
    G.add_edges_from((u, v) for u in G for v in G
                     if u < v and max(abs(u[0] - v[0]), abs(u[1] - v[1])) == 1)
    return G


def create_1_1_pattern():
    # Two 1s that both see only the same covered cell: it is their mine.
    return pattern_from_drawing("""
        1 1
        M .
    """)

def create_1_1_variants():
    # One 1 sees only the shared cell, so it is the mine; the other 1 sees it and one
    # more, which is then safe, whether or not it touches the mine.
    return (pattern_from_drawing("""
        1 1 .
        M . S
    """),)

def create_1_2_1_pattern():
    # 1-2-1 boxed in: the three clues see exactly the three cells beside them.
    # The mines are beside the 1s and the cell beside the 2 is safe.
    return pattern_from_drawing("""
        1 2 1
        M S M
    """)

def create_1_2_1_variants():
    # The same with the covered row open at one end, and at both ends: the cells
    # beyond the 1s are safe as well.
    return (pattern_from_drawing("""
        M S M S
        1 2 1 .
    """), pattern_from_drawing("""
        S M S M S
        . 1 2 1 .
    """))

def create_edge_2_pattern():
    # A 3 that sees exactly three covered cells, all mines, beside a 2 that sees two
    # of them and one more: the 2 already has both its mines, so that cell is safe.
    return pattern_from_drawing("""
        . S
        2 M
        3 M
        . M
    """)

def create_edge_pattern():
    # 1-1: the first 1 sees two cells, the second 1 sees the same two and one more;
    # the first 1's mine lies in the shared pair, so the extra cell is safe.
    return pattern_from_drawing("""
        1 ?
        1 ?
        . S
    """)

def create_2_3_2_wall():
    # 2-3-2 along a line of covered cells: the 3 sees exactly three, all mines, so
    # each 2 has its two mines and its outer cell is safe.
    return pattern_from_drawing("""
        . S
        2 M
        3 M
        2 M
        . S
    """)

def create_false_edge_5():
    # A 5 that sees exactly five covered cells: all are mines.
    return pattern_from_drawing("""
        M 5 M
        M M M
    """)

def create_1_2_pattern():
    # 1-2: the 2 needs one more mine than the 1 and sees one cell the 1 does not, so
    # that cell is a mine; the 1's mine is then in the shared pair, so the 1's own
    # outer cell is safe.
    return pattern_from_drawing("""
        . 1 2 .
        S ? ? M
    """)

def create_1_2_variants():
    # The same with the 1 at a closed end: the 2's outer cell is still a mine.
    return (pattern_from_drawing("""
        1 2 .
        ? ? M
    """),)

def create_2_2_pattern():
    # Two 2s that both see only the same two covered cells: both are mines.
    return pattern_from_drawing("""
        2 2
        M M
    """)

def create_2_2_variants():
    # One 2 sees only the shared pair, so both are mines; the other 2 sees them and
    # one more, which is then safe.
    return (pattern_from_drawing("""
        2 M
        2 M
        . S
    """),)

def create_corner_1_pattern():
    # A clue with one mine still to find and one covered neighbour: that neighbour is
    # the mine. The graph has no corner, so this matches anywhere, in any direction.
    return pattern_from_drawing("""
        1 .
        . M
    """)

def create_1_1_1_pattern():
    # Three 1s: the last sees only one covered cell, its mine, which also satisfies
    # the first, so the first 1's other cell is safe.
    return pattern_from_drawing("""
        1 1 1
        S M .
    """)

def create_triangle_1_1_1_pattern():
    # Three 1s around the fourth cell of a 2x2 square, each seeing only that cell:
    # it is the mine.
    return pattern_from_drawing("""
        1 1
        1 M
    """)


PATTERNS = {
    "1_1_shared_mine": {'graph': create_1_1_pattern(), 'variants': create_1_1_variants(), 'priority': 1},
    "1_2_1": {'graph': create_1_2_1_pattern(), 'variants': create_1_2_1_variants(), 'priority': 2},
    "edge_2": {'graph': create_edge_2_pattern(), 'priority': 3},
    "edge_pattern": {'graph': create_edge_pattern(), 'priority': 4},
    "2_3_2_wall": {'graph': create_2_3_2_wall(), 'priority': 5},
    "false_edge_5": {'graph': create_false_edge_5(), 'priority': 6},
    "1_2": {'graph': create_1_2_pattern(), 'variants': create_1_2_variants(), 'priority': 7},
    "2_2": {'graph': create_2_2_pattern(), 'variants': create_2_2_variants(), 'priority': 8},
    "corner_1": {'graph': create_corner_1_pattern(), 'priority': 9},
    "1_1_1": {'graph': create_1_1_1_pattern(), 'priority': 10},
    "triangle_1_1_1": {'graph': create_triangle_1_1_1_pattern(), 'priority': 11},
}
