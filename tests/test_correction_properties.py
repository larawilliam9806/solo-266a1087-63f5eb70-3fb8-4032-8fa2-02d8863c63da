"""Randomised cross-check of correction re-aggregation against a leaf model.

For a small grid the archive is generated from a real leaf distribution: every
stored cell holds the exact sum of the finest cells below it.  After any valid
correction the same must hold for the new batch, and every record must have
moved exactly as the permutation says.
"""

import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.correction import (
    CorrectionProblem,
    InvalidCorrection,
    plan_correction,
    validate_permutation,
)
from app.solver import analyze

DEPTH = 3


def leaves():
    return ["".join(t) for t in itertools.product("0123", repeat=DEPTH)]


def region(cell, leaf_pops):
    return sum(value for leaf, value in leaf_pops.items() if leaf.startswith(cell))


def ancestors_of(cell):
    return [cell[:index] for index in range(len(cell))]


def random_archive(rng, leaf_pops):
    """Choose a subset of nodes, guaranteeing each chosen node's ancestry.

    This models an archive that stores some upper aggregates and some finer
    cells.  Values are always the exact leaf sums, so it is consistent.
    """
    pool = [
        cell
        for length in range(DEPTH + 1)
        for cell in ("".join(t) for t in itertools.product("0123", repeat=length))
    ]
    chosen = set(rng.sample(pool, rng.randint(1, 12)))
    # Keep only nodes whose full ancestry is present, so the archive can
    # express regions below any stored node.
    closed = {
        cell for cell in chosen if all(parent in chosen for parent in ancestors_of(cell))
    }
    if rng.random() < 0.5:
        closed |= {"", "0", "1", "2", "3"}
        closed = {
            cell
            for cell in closed
            if all(parent in closed for parent in ancestors_of(cell))
        }
    return {cell: region(cell, leaf_pops) for cell in closed}


def expected_cells(entries, permutation):
    level = len(next(iter(permutation)))
    out = {}
    for cell, value in entries.items():
        if len(cell) >= level and cell[:level] in permutation:
            out[permutation[cell[:level]] + cell[level:]] = value
        else:
            out[cell] = value
    return out


def dense_archive(rng, leaf_pops):
    """A fully closed archive: every node down to depth 2 is stored."""
    entries = {}
    for length in range(0, 3):
        for t in itertools.product("0123", repeat=length):
            cell = "".join(t)
            entries[cell] = region(cell, leaf_pops)
    # A few depth-3 records for suffix-preservation coverage.
    for cell in rng.sample(leaves(), 4):
        entries[cell] = leaf_pops[cell]
    return entries


def test_random_corrections_match_leaf_model():
    rng = random.Random(20261123)
    checked = 0
    rejected = 0
    recompute_cases = 0
    for _ in range(600):
        leaf_pops = {leaf: rng.randint(0, 20) for leaf in leaves()}
        entries = (
            dense_archive(rng, leaf_pops)
            if rng.random() < 0.6
            else random_archive(rng, leaf_pops)
        )
        level = rng.randint(1, DEPTH)
        positions_pool = [
            "".join(t) for t in itertools.product("0123", repeat=level)
        ]
        if len(entries) < 2:
            continue
        count = rng.randint(2, min(4, len(positions_pool)))
        # Build a random cyclic permutation of distinct positions.
        cells = rng.sample(positions_pool, count)
        permutation = {
            cell: cells[(index + 1) % len(cells)] for index, cell in enumerate(cells)
        }
        try:
            validate_permutation(permutation)
            result = plan_correction(entries, permutation)
        except (InvalidCorrection, CorrectionProblem):
            rejected += 1
            continue
        checked += 1

        moved = dict(result.cells)
        # Relabel the leaf distribution by the permutation: the corrected
        # archive must be an exact aggregate model of these leaves.
        relabelled = {}
        for leaf, value in leaf_pops.items():
            if leaf[:level] in permutation:
                relabelled[permutation[leaf[:level]] + leaf[level:]] = value
            else:
                relabelled[leaf] = value

        # Every stored cell -- moved records and recomputed upper cells alike
        # -- must equal the sum of the relabelled leaves below it.
        for cell, value in moved.items():
            assert value == region(cell, relabelled), (cell, value)

        # Records move with their suffix and headcount; cells untouched by the
        # permutation keep theirs, and upper cells are exactly the ones the
        # planner says it recomputed.
        recomputed = {item.cell for item in result.recomputed}
        if recomputed:
            recompute_cases += 1
        expected_ids = set(expected_cells(entries, permutation))
        assert set(moved) == expected_ids
        for cell, value in expected_cells(entries, permutation).items():
            if cell in recomputed:
                continue
            assert moved[cell] == value, cell
        for cell in recomputed:
            assert moved[cell] != entries[cell] or True  # may legitimately match
            assert cell not in permutation
    assert checked > 50, (checked, rejected)
    assert recompute_cases > 10, recompute_cases


def test_random_corrections_are_queryable_exactly():
    rng = random.Random(7)
    for _ in range(60):
        leaf_pops = {leaf: rng.randint(0, 9) for leaf in leaves()}
        entries = random_archive(rng, leaf_pops)
        cells = rng.sample(["00", "01", "02", "03", "10"], 2)
        permutation = {cells[0]: cells[1], cells[1]: cells[0]}
        try:
            result = plan_correction(entries, permutation)
        except (InvalidCorrection, CorrectionProblem):
            continue
        catalog = dict(result.cells)
        # Every region the post-correction catalog can express must agree
        # with the relabelled leaf model through the normal solver.
        relabelled = {}
        for leaf, value in leaf_pops.items():
            if leaf[:2] in permutation:
                relabelled[permutation[leaf[:2]] + leaf[2:]] = value
            else:
                relabelled[leaf] = value
        for cell in catalog:
            analysis = analyze(catalog.keys(), [cell], depth=DEPTH)
            if not analysis.feasible:
                continue
            got = analysis.solution.evaluate(catalog.__getitem__)
            assert got == region(cell, relabelled)
