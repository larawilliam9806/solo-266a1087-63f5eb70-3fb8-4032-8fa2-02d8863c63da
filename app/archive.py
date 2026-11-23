"""SQLite archive storage for census batches.

Each statistical period owns exactly one batch.  A batch is stored in a single
SQLite transaction; replacing a batch deletes the previous rows and inserts the
new ones atomically, so a rejected submission leaves the previous batch intact.

Every successful submission -- initial, replacement or correction -- receives a
fresh *batch id* (批号).  A correction must name the batch id it was prepared
against; if the current id differs (a replacement happened meanwhile) the
correction is rejected and the current batch keeps serving reads.

Schema
------
``batches(period TEXT PRIMARY KEY, created_at TEXT NOT NULL, batch_id TEXT NOT NULL)``
``cells(period TEXT, cell TEXT, population INTEGER, PRIMARY KEY(period, cell))``

``cell`` is a quadtree id (root ``""``, children append ``0``-``3``); an empty
id is legal and stored as the empty string.

Databases written before batch ids existed are upgraded in place on first
connect: the column is added and existing batches get an id.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from .correction import CorrectionPlan, plan_correction

MAX_DEPTH = 12
DIGITS = "0123"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    period     TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    batch_id   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cells (
    period     TEXT    NOT NULL,
    cell       TEXT    NOT NULL,
    population INTEGER NOT NULL,
    PRIMARY KEY (period, cell),
    FOREIGN KEY (period) REFERENCES batches(period) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS cells_by_period ON cells(period);
"""


def new_batch_id() -> str:
    """Opaque, unique identifier for one immutable batch revision."""
    return uuid.uuid4().hex


class ArchiveError(Exception):
    """Base class for archive-level failures."""


class UnknownPeriod(ArchiveError):
    def __init__(self, period: str) -> None:
        super().__init__(f"no archive batch for period {period!r}")
        self.period = period


class BatchConflict(ArchiveError):
    def __init__(self, period: str) -> None:
        super().__init__(f"archive batch for period {period!r} already exists")
        self.period = period


class InvalidBatch(ArchiveError):
    pass


class BatchChanged(ArchiveError):
    """The batch id the caller prepared against is no longer current."""

    def __init__(self, period: str, expected: str, current: Optional[str]) -> None:
        super().__init__(
            f"period {period!r} is now at batch {current!r}, "
            f"not the seen batch {expected!r}"
        )
        self.period = period
        self.expected = expected
        self.current = current


class MissingCells(ArchiveError):
    def __init__(self, period: str, cells: Sequence[str]) -> None:
        super().__init__(
            f"cells {list(cells)[:5]!r} are absent from period {period!r}"
        )
        self.period = period
        self.cells = tuple(cells)


def _ensure_batch_id_column(conn: sqlite3.Connection) -> None:
    """Upgrade a pre-batch-id database in place.

    ``CREATE TABLE IF NOT EXISTS`` leaves an old ``batches`` table untouched, so
    a missing ``batch_id`` column is detected and added, with every existing
    batch receiving a fresh id.  The migration commits on its own before the
    caller's transaction starts, so old databases work with no extra step.
    """
    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(batches)")
    }
    if not columns or "batch_id" in columns:
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(batches)")
        }
        if "batch_id" not in columns:
            conn.execute(
                "ALTER TABLE batches ADD COLUMN batch_id TEXT NOT NULL DEFAULT ''"
            )
        rows = conn.execute(
            "SELECT period FROM batches WHERE batch_id = ''"
        ).fetchall()
        for row in rows:
            conn.execute(
                "UPDATE batches SET batch_id = ? WHERE period = ?",
                (new_batch_id(), row["period"]),
            )
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    conn.execute("COMMIT")


def validate_cell(cell: str, depth: int = MAX_DEPTH) -> str:
    if not isinstance(cell, str):
        raise InvalidBatch("cell id must be a string")
    if len(cell) > depth:
        raise InvalidBatch(
            f"cell id {cell!r} has {len(cell)} digits, maximum is {depth}"
        )
    if any(digit not in DIGITS for digit in cell):
        raise InvalidBatch(f"cell id {cell!r} contains a digit outside 0-3")
    return cell


@dataclass(frozen=True)
class BatchInfo:
    period: str
    cell_count: int
    total_population: int
    created_at: str
    batch_id: str


@dataclass(frozen=True)
class CorrectionResult:
    period: str
    batch_id: str
    seen_batch_id: str
    created_at: str
    cell_count: int
    total_population: int
    permutation: Tuple[Tuple[str, str], ...]
    moves: Tuple[Tuple[str, str, int], ...]
    recomputed: Tuple[Tuple[str, int, int], ...]


class Archive:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connect(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.isolation_level = None
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA busy_timeout = 30000")
            if write:
                # WAL is persisted in the database file; readers then see a
                # consistent snapshot without blocking the writer.
                conn.execute("PRAGMA journal_mode = WAL")
            # DDL is idempotent and must run outside an explicit transaction.
            conn.executescript(_SCHEMA)
            _ensure_batch_id_column(conn)
            # BEGIN IMMEDIATE serializes writers, so the existence check and
            # the batch swap cannot interleave with a concurrent submission.
            conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()

    def replace_batch(
        self,
        period: str,
        entries: Sequence[Tuple[str, int]],
        *,
        allow_replace: bool = False,
    ) -> BatchInfo:
        """Atomically store a whole batch for ``period``.

        Duplicate cell ids in ``entries`` are rejected.  When ``allow_replace``
        is false an existing batch raises :class:`BatchConflict`; in either case
        the operation is a single transaction, so failures keep the old batch.
        """
        if not period:
            raise InvalidBatch("statistical period must be a non-empty string")
        seen: Dict[str, int] = {}
        for cell, population in entries:
            validate_cell(cell)
            if cell in seen:
                raise InvalidBatch(f"duplicate cell id {cell!r} in batch")
            if isinstance(population, bool) or not isinstance(population, int):
                raise InvalidBatch(f"population for cell {cell!r} must be an integer")
            if population < 0:
                raise InvalidBatch(f"population for cell {cell!r} must be non-negative")
            seen[cell] = population

        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        batch_id = new_batch_id()
        with self._connect(write=True) as conn:
            existing = conn.execute(
                "SELECT 1 FROM batches WHERE period = ?", (period,)
            ).fetchone()
            if existing is not None and not allow_replace:
                raise BatchConflict(period)
            # One transaction: delete, upsert batch row, insert new cells.
            conn.execute("DELETE FROM cells WHERE period = ?", (period,))
            conn.execute(
                "INSERT INTO batches(period, created_at, batch_id) VALUES(?, ?, ?) "
                "ON CONFLICT(period) DO UPDATE SET "
                "created_at = excluded.created_at, "
                "batch_id = excluded.batch_id",
                (period, created_at, batch_id),
            )
            conn.executemany(
                "INSERT INTO cells(period, cell, population) VALUES(?, ?, ?)",
                [(period, cell, pop) for cell, pop in seen.items()],
            )
        return BatchInfo(
            period=period,
            cell_count=len(seen),
            total_population=sum(seen.values()),
            created_at=created_at,
            batch_id=batch_id,
        )

    def correct_batch(
        self,
        period: str,
        seen_batch_id: str,
        permutation: Mapping[str, str],
    ) -> CorrectionResult:
        """Apply a same-level cell-id permutation as one atomic new revision.

        The correction is prepared from a read snapshot of the current batch.
        The write transaction then re-checks the batch id under
        ``BEGIN IMMEDIATE``: if a concurrent replacement or correction landed
        in between, the correction is rejected with :class:`BatchChanged` and
        the current batch keeps serving reads unchanged.

        All validation and re-aggregation failures likewise leave the current
        batch untouched.  Success mints a fresh batch id.
        """
        if not period:
            raise InvalidBatch("statistical period must be a non-empty string")
        if not isinstance(seen_batch_id, str) or not seen_batch_id:
            raise InvalidBatch("seen_batch_id must be a non-empty string")

        with self.snapshot(period) as snapshot:
            snapshot_batch_id = snapshot.batch_id()
            if snapshot_batch_id != seen_batch_id:
                raise BatchChanged(period, seen_batch_id, snapshot_batch_id)
            entries = snapshot.entries()
            plan: CorrectionPlan = plan_correction(entries, permutation)

        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        batch_id = new_batch_id()
        with self._connect(write=True) as conn:
            row = conn.execute(
                "SELECT batch_id FROM batches WHERE period = ?", (period,)
            ).fetchone()
            if row is None:
                raise UnknownPeriod(period)
            if row["batch_id"] != seen_batch_id:
                raise BatchChanged(period, seen_batch_id, row["batch_id"])
            # Rewrite the whole batch from the plan inside this transaction.
            conn.execute("DELETE FROM cells WHERE period = ?", (period,))
            conn.execute(
                "UPDATE batches SET created_at = ?, batch_id = ? WHERE period = ?",
                (created_at, batch_id, period),
            )
            conn.executemany(
                "INSERT INTO cells(period, cell, population) VALUES(?, ?, ?)",
                [(period, cell, pop) for cell, pop in plan.cells],
            )
        return CorrectionResult(
            period=period,
            batch_id=batch_id,
            seen_batch_id=seen_batch_id,
            created_at=created_at,
            cell_count=plan.cell_count,
            total_population=plan.total_population,
            permutation=plan.permutation,
            moves=tuple(
                (move.source, move.target, move.population) for move in plan.moves
            ),
            recomputed=tuple(
                (item.cell, item.previous_population, item.population)
                for item in plan.recomputed
            ),
        )

    def batch_info(self, period: str) -> BatchInfo:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT b.created_at, b.batch_id, "
                "       COUNT(c.cell) AS n, "
                "       COALESCE(SUM(c.population), 0) AS total "
                "FROM batches b LEFT JOIN cells c ON c.period = b.period "
                "WHERE b.period = ? GROUP BY b.period",
                (period,),
            ).fetchone()
        if row is None:
            raise UnknownPeriod(period)
        return BatchInfo(
            period=period,
            cell_count=row["n"],
            total_population=row["total"],
            created_at=row["created_at"],
            batch_id=row["batch_id"],
        )

    def exists(self, period: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM batches WHERE period = ?", (period,)
            ).fetchone()
        return row is not None

    def catalog(self, period: str) -> List[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM batches WHERE period = ?", (period,)
            ).fetchone()
            if row is None:
                raise UnknownPeriod(period)
            rows = conn.execute(
                "SELECT cell FROM cells WHERE period = ? ORDER BY length(cell), cell",
                (period,),
            ).fetchall()
        return [r["cell"] for r in rows]

    def populations(self, period: str, cells: Iterable[str]) -> Dict[str, int]:
        """Read populations for exactly the requested cells of one batch."""
        wanted = list(dict.fromkeys(cells))
        if not wanted:
            return {}
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM batches WHERE period = ?", (period,)
            ).fetchone()
            if row is None:
                raise UnknownPeriod(period)
            found = _fetch_populations(conn, period, wanted)
        _require_all(period, wanted, found)
        return found

    @contextmanager
    def snapshot(self, period: str) -> Iterator["BatchSnapshot"]:
        """Consistent read view of one batch.

        The catalog and the populations fetched through the returned snapshot
        always come from the same transaction, even if a writer replaces the
        batch concurrently.
        """
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.isolation_level = None
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA busy_timeout = 30000")
            conn.executescript(_SCHEMA)
            _ensure_batch_id_column(conn)
            conn.execute("BEGIN")
            row = conn.execute(
                "SELECT 1 FROM batches WHERE period = ?", (period,)
            ).fetchone()
            if row is None:
                raise UnknownPeriod(period)
            yield BatchSnapshot(conn, period)
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()


class BatchSnapshot:
    """Read-only consistent view of one period used by the solver."""

    def __init__(self, conn: sqlite3.Connection, period: str) -> None:
        self._conn = conn
        self.period = period

    def batch_id(self) -> str:
        row = self._conn.execute(
            "SELECT batch_id FROM batches WHERE period = ?", (self.period,)
        ).fetchone()
        return row["batch_id"]

    def catalog(self) -> List[str]:
        rows = self._conn.execute(
            "SELECT cell FROM cells WHERE period = ? ORDER BY length(cell), cell",
            (self.period,),
        ).fetchall()
        return [r["cell"] for r in rows]

    def entries(self) -> Dict[str, int]:
        rows = self._conn.execute(
            "SELECT cell, population FROM cells WHERE period = ? "
            "ORDER BY length(cell), cell",
            (self.period,),
        ).fetchall()
        return {r["cell"]: r["population"] for r in rows}

    def populations(self, cells: Iterable[str]) -> Dict[str, int]:
        wanted = list(dict.fromkeys(cells))
        if not wanted:
            return {}
        found = _fetch_populations(self._conn, self.period, wanted)
        _require_all(self.period, wanted, found)
        return found


def _fetch_populations(
    conn: sqlite3.Connection, period: str, wanted: Sequence[str]
) -> Dict[str, int]:
    found: Dict[str, int] = {}
    # Stay under SQLite's variable limit for large expressions.
    for start in range(0, len(wanted), 500):
        chunk = wanted[start : start + 500]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT cell, population FROM cells "
            f"WHERE period = ? AND cell IN ({placeholders})",
            (period, *chunk),
        ).fetchall()
        found.update({r["cell"]: r["population"] for r in rows})
    return found


def _require_all(period: str, wanted: Sequence[str], found: Dict[str, int]) -> None:
    missing = [cell for cell in wanted if cell not in found]
    if missing:
        raise MissingCells(period, missing)
