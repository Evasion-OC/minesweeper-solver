"""Exact mine probabilities for the guess, checked against counting every layout."""

import random
from fractions import Fraction
from itertools import combinations

import pytest

import minesweeper.probability as probability
from minesweeper.deductive import (
    FLAGGED,
    UNKNOWN,
    InconsistentObservation,
    Observation,
    RegionCache,
    choose_action,
    deduce_all,
)
from minesweeper.probability import TooManyLayouts, mine_probabilities
from minesweeper.symmetry import grid_transforms
from test_canonical import some_layout_fits, transformed
from test_regions import neighbours, positions_from_play

U = UNKNOWN


def counted(observation):
    """Each covered cell's share of the layouts that fit, by listing every layout."""
    n_rows, n_cols, cells = observation.n_rows, observation.n_cols, observation.cells
    unknown = [i for i, s in enumerate(cells) if s == UNKNOWN]
    flags = {i for i, s in enumerate(cells) if s == FLAGGED}
    hits, fitting = dict.fromkeys(unknown, 0), 0
    for layout in combinations(unknown, observation.total_mines - len(flags)):
        mines = set(layout) | flags
        if all(sum(n in mines for n in neighbours(i, n_rows, n_cols)) == s for i, s in enumerate(cells) if s >= 0):
            fitting += 1
            for i in layout:
                hits[i] += 1
    return {i: Fraction(h, fitting) for i, h in hits.items()}


def small_observations():
    """Positions from play and boards seen from random layouts with some flags."""
    for args in [(4, 4, 3, 60, 1), (5, 5, 4, 60, 2), (5, 5, 6, 40, 3), (4, 6, 5, 40, 4)]:
        yield from positions_from_play(*args)
    rng = random.Random(9)
    for _ in range(600):
        n_rows, n_cols = rng.choice([(3, 3), (3, 4), (4, 4), (2, 6)])
        size = n_rows * n_cols
        mines = set(rng.sample(range(size), rng.randrange(size // 2)))
        cells = tuple(rng.choice([U, U, FLAGGED]) if i in mines else
                      rng.choice([U, sum(n in mines for n in neighbours(i, n_rows, n_cols))]) for i in range(size))
        yield Observation(n_rows, n_cols, len(mines), cells)


def test_probabilities_equal_the_share_of_layouts():
    cache = RegionCache()
    checked = 0
    for observation in small_observations():
        expected = counted(observation)
        assert mine_probabilities(observation) == mine_probabilities(observation, cache) == expected
        checked += 1
    assert checked > 1000


def test_probabilities_agree_with_the_forced_moves():
    # Probability 0 or 1 exactly where SAT proves the cell safe or a mine.
    for observation in small_observations():
        found = mine_probabilities(observation)
        certain = {("reveal", i) for i, p in found.items() if p == 0} | {("flag", i) for i, p in found.items() if p == 1}
        forced = {(a.kind, a.cell[0] * observation.n_cols + a.cell[1]) for a in deduce_all(observation)}
        assert certain == forced
        assert sum(found.values()) == observation.total_mines - observation.cells.count(FLAGGED)


def test_probabilities_follow_the_board_under_its_symmetries():
    for observation in positions_from_play(8, 8, 10, games=5, seed=12):
        found = mine_probabilities(observation)
        for transform in grid_transforms(8, 8):
            moved = mine_probabilities(transformed(observation, transform))
            assert moved == {transform.mapping[i]: p for i, p in found.items()}


def test_the_guess_is_the_least_likely_cell():
    guesses = 0
    for observation in positions_from_play(8, 8, 12, games=40, seed=13):  # dense boards need guesses
        action = choose_action(observation)
        if action.reason == "guess":
            guesses += 1
            found = mine_probabilities(observation)
            chosen = action.cell[0] * observation.n_cols + action.cell[1]
            assert found[chosen] == min(found.values())
            assert chosen == min(i for i, p in found.items() if p == found[chosen])  # ties go to the lowest index
    assert guesses > 30


def test_a_region_too_large_to_count_falls_back_to_a_cell_no_clue_touches(monkeypatch):
    def too_many(observation, cache=None, cap=0):
        raise TooManyLayouts("forced for the test")
    monkeypatch.setattr(probability, "mine_probabilities", too_many)
    # A 1 sees two cells, the rest of the row is untouched: no move is forced.
    board = Observation(1, 6, 2, (U, 1, U, U, U, U))
    assert choose_action(board).cell == (0, 3)


def test_with_every_covered_cell_touching_a_clue_the_fallback_is_the_lowest_cell(monkeypatch):
    def too_many(observation, cache=None, cap=0):
        raise TooManyLayouts("forced for the test")
    monkeypatch.setattr(probability, "mine_probabilities", too_many)
    board = Observation(1, 3, 1, (U, 1, U))  # one region, no free cell, nothing forced
    assert choose_action(board).cell == (0, 0)


def test_a_region_too_deep_to_count_is_not_counted():
    # A row of 1s under 1,700 covered cells, each seen by a different set of clues: far more
    # groups than the guard allows. A mine in every third column fits it.
    width = 1700
    board = Observation(2, width, len(range(1, width, 3)), tuple([U] * width) + tuple([1] * width))
    with pytest.raises(TooManyLayouts):
        mine_probabilities(board)


def test_layout_counts_are_reused_in_any_orientation():
    for observation in positions_from_play(8, 8, 12, games=4, seed=14):
        cache = RegionCache()
        found = mine_probabilities(observation, cache)
        stored = len(cache._counts)
        for transform in grid_transforms(8, 8):
            assert mine_probabilities(transformed(observation, transform), cache) == {
                transform.mapping[i]: p for i, p in found.items()}
        assert len(cache._counts) == stored  # no region was counted again
    # choose_action keeps its counts in the cache it is given.
    guess = next(o for o in positions_from_play(8, 8, 12, games=40, seed=13) if choose_action(o).reason == "guess")
    cache = RegionCache()
    choose_action(guess, cache)
    assert len(cache._counts) > 0


def test_the_cap_stops_a_large_count():
    board = Observation(2, 3, 2, (U, U, U, 1, 2, 1))
    with pytest.raises(TooManyLayouts):
        mine_probabilities(board, cap=1)
    assert mine_probabilities(board) == {0: 1, 1: 0, 2: 1}


def test_a_board_no_layout_fits_is_rejected():
    rng = random.Random(3)
    rejected = 0
    for _ in range(1500):
        n_rows, n_cols = rng.choice([(1, 3), (2, 2), (2, 3), (3, 3)])
        cells = tuple(rng.choice([U, U, U, FLAGGED, 0, 1, 1, 2, 3]) for _ in range(n_rows * n_cols))
        observation = Observation(n_rows, n_cols, rng.randrange(n_rows * n_cols + 1), cells)
        if not some_layout_fits(observation):
            rejected += 1
            with pytest.raises(InconsistentObservation):
                mine_probabilities(observation)
    assert rejected > 500
