"""Cell-id transpositions and exact re-aggregation.

A correction is a *permutation* of 2-8 same-level cell ids.  Every archived
record that lives at or below an old position moves to the matching new
position with its suffix and headcount untouched::

    old position p  ->  new position q
    record p + suffix  ->  record q + suffix

The permutation may be a transposition or a longer cycle.

Stored upper-level cells
------------------------
Archived cells are aggregates over the finest cells below them.  Relabelling a
subtree keeps that subtree's own records intact, but an ancestor whose *set of
source regions* changes must be re-aggregated.  For an ancestor ``u`` the new
region is::

    region(u) - union(regions leaving u) + union(regions entering u)

Both the outgoing and incoming region populations are derived from the
**pre-correction** batch.  When a position is not stored individually the exact
value is obtained with :func:`app.solver.analyze`, which only visits submitted
ids and their prefixes -- no finest cell is ever enumerated.

If the set of source regions does not change (the classic case: swapping
siblings below the same parent) the aggregate is left untouched.  Such
corrections therefore work even when the individual district headcounts cannot
be isolated from the archive.

Failures are reported against the *upper-level* cell: a region that the
pre-correction archive cannot express exactly is ``missing``; an aggregate
whose re-computation would be negative is ``negative``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .solver import DIGITS, MAX_DEPTH, analyze

MIN_POSITIONS = 2
MAX_POSITIONS = 8


class InvalidCorrection(Exception):
    """The requested permutation is malformed or out of range."""


class CorrectionProblem(Exception):
    """The pre-correction archive cannot support the requested correction.

    ``missing_cells`` and ``negative_cells`` name the stored *upper-level*
    cells whose re-aggregation failed, as required by the correction report.
    """

    def __init__(
        self,
        missing_cells: Iterable[str] = (),
        negative_cells: Iterable[str] = (),
        missing_regions: Iterable[str] = (),
    ) -> None:
        self.missing_cells: Tuple[str, ...] = tuple(missing_cells)
        self.negative_cells: Tuple[str, ...] = tuple(negative_cells)
        self.missing_regions: Tuple[str, ...] = tuple(missing_regions)
        self.upper_cells: Tuple[str, ...] = tuple(
            dict.fromkeys(
                list(self.missing_cells) + list(self.negative_cells)
            )
        )
        parts: List[str] = []
        if self.missing_cells:
            parts.append(
                "archive missing for upper cells "
                + repr(list(self.missing_cells))
            )
        if self.negative_cells:
            parts.append(
                "correction would make upper cells negative "
                + repr(list(self.negative_cells))
            )
        super().__init__("; ".join(parts) or "correction is not possible")


@dataclass(frozen=True)
class Move:
    source: str
    target: str
    population: int


@dataclass(frozen=True)
class Recompute:
    cell: str
    previous_population: int
    population: int


@dataclass(frozen=True)
class CorrectionPlan:
    """Fully resolved, validated correction ready to be written atomically."""

    permutation: Tuple[Tuple[str, str], ...]
    cells: Tuple[Tuple[str, int], ...]
    moves: Tuple[Move, ...]
    recomputed: Tuple[Recompute, ...]

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    @property
    def total_population(self) -> int:
        return sum(population for _, population in self.cells)


def validate_position(cell: str, depth: int = MAX_DEPTH) -> str:
    if not isinstance(cell, str):
        raise InvalidCorrection("cell id must be a string")
    if len(cell) > depth:
        raise InvalidCorrection(
            f"cell id {cell!r} has {len(cell)} digits, maximum is {depth}"
        )
    if any(digit not in DIGITS for digit in cell):
        raise InvalidCorrection(
            f"cell id {cell!r} contains a digit outside 0-3"
        )
    return cell


def validate_permutation(
    permutation: Mapping[str, str], depth: int = MAX_DEPTH
) -> Dict[str, str]:
    """Return a copy of a well-formed same-level permutation.

    A correction must name between 2 and 8 distinct cells, all at the same
    level, and map them one-to-one onto their own positions.
    """
    if not isinstance(permutation, Mapping):
        raise InvalidCorrection("correction must map old cell ids to new cell ids")
    result: Dict[str, str] = {}
    for source, target in permutation.items():
        validate_position(source, depth)
        validate_position(target, depth)
        if source in result:
            raise InvalidCorrection(f"duplicate cell id {source!r} in correction")
        result[source] = target

    if len(result) < MIN_POSITIONS:
        raise InvalidCorrection(
            f"a correction needs at least {MIN_POSITIONS} cell ids"
        )
    if len(result) > MAX_POSITIONS:
        raise InvalidCorrection(
            f"a correction accepts at most {MAX_POSITIONS} cell ids"
        )
    if len(set(result.values())) != len(result):
        raise InvalidCorrection("corrected cells must map one-to-one")
    if set(result.values()) != set(result):
        raise InvalidCorrection(
            "corrected cells must move among their own positions"
        )
    levels = {len(cell) for cell in result}
    if len(levels) != 1:
        raise InvalidCorrection("corrected cells must all be on the same level")
    if all(source == target for source, target in result.items()):
        raise InvalidCorrection("a correction must move at least one cell")
    return result


def _region_population(
    cell: str,
    catalog: Mapping[str, int],
    cache: Dict[str, Optional[int]],
    missing: Optional[List[str]] = None,
) -> Optional[int]:
    """Exact population of ``cell``'s region in the pre-correction archive.

    ``None`` means the archive cannot express the region exactly.  The solver
    only visits submitted ids and their prefixes, never a finest cell.  Any
    diagnostic blocks are appended to ``missing``.
    """
    if cell in cache:
        value = cache[cell]
    elif cell in catalog:
        value = catalog[cell]
    else:
        analysis = analyze(catalog.keys(), [cell], depth=MAX_DEPTH)
        value = (
            analysis.solution.evaluate(catalog.__getitem__)
            if analysis.feasible
            else None
        )
        if value is None and missing is not None:
            missing.extend(analysis.missing_blocks)
    cache[cell] = value
    return value


def plan_correction(
    entries: Mapping[str, int],
    permutation: Mapping[str, str],
    *,
    depth: int = MAX_DEPTH,
) -> CorrectionPlan:
    """Resolve a permutation into the complete post-correction batch.

    Raises :class:`InvalidCorrection` for a malformed permutation and
    :class:`CorrectionProblem` when an upper-level aggregate cannot be
    recomputed exactly from the pre-correction archive.
    """
    catalog: Dict[str, int] = {cell: int(pop) for cell, pop in entries.items()}
    perm = validate_permutation(permutation, depth=depth)
    positions = tuple(sorted(perm))
    level = len(positions[0])
    position_set = set(positions)

    # A record moves when its first ``level`` digits name a corrected cell.
    moved_rows: Dict[str, int] = {}
    for cell in catalog:
        if len(cell) >= level and cell[:level] in position_set:
            moved_rows[cell] = catalog[cell]

    # Stationary records keep their ids unchanged.
    new_cells: Dict[str, int] = {
        cell: population
        for cell, population in catalog.items()
        if cell not in moved_rows
    }

    moves: List[Move] = []
    for cell, population in sorted(
        moved_rows.items(), key=lambda item: (len(item[0]), item[0])
    ):
        target = perm[cell[:level]] + cell[level:]
        if target in new_cells:
            # Cannot happen for a validated permutation, but never overwrite
            # an unrelated record silently.
            raise InvalidCorrection(
                f"corrected cell {cell!r} would collide at {target!r}"
            )
        new_cells[target] = population
        if target != cell:
            moves.append(Move(source=cell, target=target, population=population))

    # Only stored strict ancestors of a corrected position can have their
    # coverage sources changed.  All positions share one level, so at most
    # level * 8 distinct ancestors need inspection.
    ancestor_candidates = {
        position[:index]
        for position in positions
        for index in range(level)
    }

    region_cache: Dict[str, Optional[int]] = {}
    missing: List[str] = []
    missing_regions: List[str] = []
    negative: List[str] = []
    recomputed: List[Recompute] = []

    for ancestor in sorted(
        ancestor_candidates, key=lambda cell: (len(cell), cell)
    ):
        if ancestor not in catalog:
            continue
        leaving = {
            position
            for position in positions
            if position.startswith(ancestor)
        }
        entering = {
            position
            for position in positions
            if perm[position].startswith(ancestor)
        }
        if leaving == entering:
            # Same original regions land in this cell; its aggregate is
            # unchanged even if individual region counts are unknown.
            continue

        delta = 0
        problem = False
        inconsistent = False
        for position in sorted(leaving - entering):
            value = _region_population(
                position, catalog, region_cache, missing_regions
            )
            if value is None:
                problem = True
                break
            if value < 0:
                # The pre-correction archive contradicts tree sums for this
                # region; the affected upper cell cannot be trusted.
                inconsistent = True
                break
            delta -= value
        if problem:
            missing.append(ancestor)
            continue
        if inconsistent:
            negative.append(ancestor)
            continue
        for position in sorted(entering - leaving):
            value = _region_population(
                position, catalog, region_cache, missing_regions
            )
            if value is None:
                problem = True
                break
            if value < 0:
                inconsistent = True
                break
            delta += value
        if problem:
            missing.append(ancestor)
            continue
        if inconsistent:
            negative.append(ancestor)
            continue

        new_value = catalog[ancestor] + delta
        if new_value < 0:
            negative.append(ancestor)
            continue
        new_cells[ancestor] = new_value
        recomputed.append(
            Recompute(
                cell=ancestor,
                previous_population=catalog[ancestor],
                population=new_value,
            )
        )

    if missing or negative:
        raise CorrectionProblem(
            missing_cells=sorted(set(missing), key=lambda c: (len(c), c)),
            negative_cells=sorted(set(negative), key=lambda c: (len(c), c)),
            missing_regions=tuple(
                dict.fromkeys(
                    sorted(set(missing_regions), key=lambda c: (len(c), c))
                )
            ),
        )

    ordered_cells = tuple(
        sorted(new_cells.items(), key=lambda item: (len(item[0]), item[0]))
    )
    return CorrectionPlan(
        permutation=tuple(sorted(perm.items())),
        cells=ordered_cells,
        moves=tuple(moves),
        recomputed=tuple(recomputed),
    )


def cycle_permutation(cells: Sequence[str]) -> Dict[str, str]:
    """Turn an ordered list of same-level ids into a cyclic permutation.

    ``["01", "02"]`` swaps the two cells; longer lists rotate the records so
    that each cell's records move to the next cell and the last moves first.
    """
    ordered = list(cells)
    if len(set(ordered)) != len(ordered):
        raise InvalidCorrection("corrected cells must be distinct")
    return {
        cell: ordered[(index + 1) % len(ordered)]
        for index, cell in enumerate(ordered)
    }
