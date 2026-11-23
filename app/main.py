"""FastAPI service exposing the census archive.

Endpoints
---------
``POST /archive``                 submit (or atomically replace) one period's batch
``GET  /archive/{period}``        batch metadata (including the current batch id)
``POST /archive/{period}/correct`` apply a same-level cell-id permutation
``POST /census/query``            minimal-read expression + union population

The census endpoint works in two clearly separated phases inside one SQLite
read snapshot: first derive the expression from the period's catalog only, then
read headcounts for exactly the cells that appear in the expression.

Corrections are optimistic: the caller submits the batch id it has seen.  A
fresh batch id is minted only when the correction commits, and the old batch
keeps serving reads when anything fails.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from .archive import (
    Archive,
    BatchChanged,
    BatchConflict,
    InvalidBatch,
    MissingCells,
    UnknownPeriod,
    validate_cell,
)
from .correction import (
    CorrectionProblem,
    InvalidCorrection,
    cycle_permutation,
    validate_permutation,
)
from .solver import MAX_DEPTH, analyze

MAX_CELLS_PER_REQUEST = 100_000
DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "archive.sqlite3"

_SEEN_BATCH_KEYS = (
    "seen_batch_id",
    "seen_batch",
    "seenBatchId",
    "batch_id",
    "batchId",
    "expected_batch_id",
    "expected_batch",
    "base_batch_id",
    "source_batch_id",
    "seen_batch_no",
    "batch_no",
    "batch_version",
    "batch",
    "version",
    "revision",
    "current_batch_id",
    "seen",
)
_PERIOD_KEYS = ("period", "statistical_period", "period_id", "periodId")
_PERMUTATION_KEYS = (
    "permutation",
    "mapping",
    "map",
    "cell_mapping",
    "cell_map",
    "cell_permutation",
    "permute",
    "correction",
    "corrections",
    "cell_correction",
    "cell_corrections",
    "cell_changes",
    "changes",
    "replacements",
)
_POSITION_KEYS = (
    "cells",
    "cell_ids",
    "cell_codes",
    "grid_cells",
    "positions",
    "ids",
    "cell_list",
    "order",
)
_SWAP_KEYS = (
    "swaps",
    "swap",
    "exchange",
    "exchanges",
    "transpositions",
    "pairs",
    "moves",
    "movements",
)
_METADATA_KEYS = set(_SEEN_BATCH_KEYS) | set(_PERIOD_KEYS) | set(_PERMUTATION_KEYS) | set(
    _POSITION_KEYS
) | set(_SWAP_KEYS) | {
    "replace",
    "reason",
    "note",
    "comment",
    "from_cells",
    "to_cells",
    "old_cells",
    "new_cells",
    "source_cells",
    "target_cells",
    "from_cell",
    "to_cell",
    "old_cell",
    "new_cell",
    "source_cell",
    "target_cell",
}


def default_database_path() -> str:
    return os.environ.get("CENSUS_DB", str(DEFAULT_DB))


def get_archive() -> Archive:
    return Archive(default_database_path())


def _check_cell(cell: str) -> str:
    try:
        return validate_cell(cell)
    except InvalidBatch as exc:
        # Re-raise as ValueError so pydantic reports it as a validation error.
        raise ValueError(str(exc)) from exc


class CellEntry(BaseModel):
    cell: str = Field(..., description='quadtree cell id; the root is ""')
    population: int = Field(..., ge=0)


class BatchSubmission(BaseModel):
    period: str = Field(..., min_length=1, description="statistical period label")
    replace: bool = Field(
        False,
        description="replace an existing batch atomically when true",
    )
    cells: List[CellEntry] = Field(
        default_factory=list, max_length=MAX_CELLS_PER_REQUEST
    )

    @field_validator("period")
    @classmethod
    def _period_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("period must not be blank")
        return value

    @field_validator("cells")
    @classmethod
    def _cells_valid(cls, value: List[CellEntry]) -> List[CellEntry]:
        seen = set()
        for entry in value:
            _check_cell(entry.cell)
            if entry.cell in seen:
                raise ValueError(f"duplicate cell id {entry.cell!r} in batch")
            seen.add(entry.cell)
        return value


class BatchResponse(BaseModel):
    period: str
    cell_count: int
    total_population: int
    created_at: str
    replaced: bool
    batch_id: str


class QueryRequest(BaseModel):
    period: str = Field(..., min_length=1)
    cells: List[str] = Field(..., max_length=MAX_CELLS_PER_REQUEST)

    @field_validator("cells")
    @classmethod
    def _cells_valid(cls, value: List[str]) -> List[str]:
        return [_check_cell(cell) for cell in value]


class ExpressionTermModel(BaseModel):
    cell: str
    coefficient: int


class CensusResponse(BaseModel):
    period: str
    cells: List[str]
    expression: str
    terms: List[ExpressionTermModel]
    retrieved_cells: int
    total_population: int


class MissingArchiveResponse(BaseModel):
    detail: str
    period: str
    missing_blocks: List[str]
    truncated: bool = False


class CorrectionMove(BaseModel):
    source: str
    target: str
    population: int


class CorrectionRecompute(BaseModel):
    cell: str
    previous_population: int
    population: int


class CorrectionResponse(BaseModel):
    period: str
    batch_id: str
    previous_batch_id: str
    seen_batch_id: str
    created_at: str
    cell_count: int
    total_population: int
    permutation: Dict[str, str]
    moves: List[CorrectionMove]
    recomputed: List[CorrectionRecompute]


class CorrectionFailureResponse(BaseModel):
    detail: str
    period: str
    seen_batch_id: Optional[str] = None
    current_batch_id: Optional[str] = None
    upper_cells: List[str] = Field(default_factory=list)
    missing_cells: List[str] = Field(default_factory=list)
    negative_cells: List[str] = Field(default_factory=list)
    missing_regions: List[str] = Field(default_factory=list)


app = FastAPI(
    title="Census Archive Service",
    description=(
        "Quadtree census archive. Submit whole statistical-period batches "
        "atomically and compute union populations with minimal-read expressions."
    ),
    version="1.0.0",
)


@app.exception_handler(UnknownPeriod)
async def _unknown_period_handler(request: Request, exc: UnknownPeriod) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc), "period": exc.period},
    )


@app.exception_handler(MissingCells)
async def _missing_cells_handler(request: Request, exc: MissingCells) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "detail": str(exc),
            "period": exc.period,
            "missing_blocks": list(exc.cells),
            "truncated": False,
        },
    )


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/archive", response_model=BatchResponse, status_code=201)
def submit_batch(
    submission: BatchSubmission,
    archive: Archive = Depends(get_archive),
) -> BatchResponse:
    try:
        info = archive.replace_batch(
            submission.period,
            [(entry.cell, entry.population) for entry in submission.cells],
            allow_replace=submission.replace,
        )
    except BatchConflict as exc:
        raise HTTPException(
            status_code=409,
            detail=f"{exc}; pass replace=true to replace it atomically",
        ) from exc
    except InvalidBatch as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return BatchResponse(
        period=info.period,
        cell_count=info.cell_count,
        total_population=info.total_population,
        created_at=info.created_at,
        replaced=submission.replace,
        batch_id=info.batch_id,
    )


@app.get("/archive/{period}", response_model=BatchResponse)
def read_batch(period: str, archive: Archive = Depends(get_archive)) -> BatchResponse:
    info = archive.batch_info(period)
    return BatchResponse(
        period=info.period,
        cell_count=info.cell_count,
        total_population=info.total_population,
        created_at=info.created_at,
        replaced=False,
        batch_id=info.batch_id,
    )


def _first(body: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in body and body[key] is not None:
            return body[key]
    return None


def _as_pairs(value: Any, *, name: str) -> List[Tuple[str, str]]:
    """Normalise ``[[old, new], ...]`` or ``{"old": "new"}`` to pairs."""
    if isinstance(value, Mapping):
        pairs = [(str(old), str(new)) for old, new in value.items()]
    elif isinstance(value, (list, tuple)):
        pairs = []
        for item in value:
            if isinstance(item, Mapping):
                old = _first(item, ("source", "from", "old", "old_cell", "cell"))
                new = _first(item, ("target", "to", "new", "new_cell", "replacement"))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                old, new = item
            else:
                raise InvalidCorrection(
                    f"{name} entries must be old/new cell pairs"
                )
            if old is None or new is None:
                raise InvalidCorrection(
                    f"{name} entries must be old/new cell pairs"
                )
            pairs.append((str(old), str(new)))
    else:
        raise InvalidCorrection(f"{name} must be a mapping or a list of pairs")

    seen = set()
    for source, _ in pairs:
        if source in seen:
            raise InvalidCorrection(f"duplicate cell id {source!r} in {name}")
        seen.add(source)
    return pairs


def _correction_permutation(body: Mapping[str, Any]) -> Dict[str, str]:
    """Read a permutation from any of the supported request shapes.

    Accepted forms (all name 2-8 same-level ids):

    * ``"permutation": {"01": "02", "02": "01"}`` -- explicit one-to-one map
    * ``"permutation": [["01", "02"], ["02", "01"]]`` -- same, as pairs
    * ``"swaps": [["01", "02"], ["03", "00"]]`` -- independent swaps
    * ``"cells": ["01", "02", "03"]`` -- cyclic rotation in the given order
    * ``"from_cells"``/``"to_cells"`` or ``"old_cells"``/``"new_cells"``
    """
    explicit = _first(body, _PERMUTATION_KEYS)
    if explicit is not None:
        return dict(_as_pairs(explicit, name="permutation"))

    swaps = _first(body, _SWAP_KEYS)
    if swaps is not None:
        # Each swap names two cells that exchange places, so it contributes
        # both directions; explicit permutations remain directional.
        pairs = _as_pairs(swaps, name="swaps")
        result: Dict[str, str] = {}
        for old, new in pairs:
            for source, target in ((old, new), (new, old)):
                if source in result and result[source] != target:
                    raise InvalidCorrection(
                        f"cell {source!r} appears in more than one swap"
                    )
                result[source] = target
        return result

    old_cells = _first(body, ("from_cells", "old_cells", "source_cells"))
    new_cells = _first(body, ("to_cells", "new_cells", "target_cells"))
    if old_cells is not None or new_cells is not None:
        if (
            not isinstance(old_cells, (list, tuple))
            or not isinstance(new_cells, (list, tuple))
            or len(old_cells) != len(new_cells)
        ):
            raise InvalidCorrection(
                "from_cells and to_cells must be equal-length lists"
            )
        return {str(old): str(new) for old, new in zip(old_cells, new_cells)}

    cells = _first(body, _POSITION_KEYS)
    if cells is not None:
        if isinstance(cells, Mapping):
            return dict(_as_pairs(cells, name="cells"))
        if not isinstance(cells, (list, tuple)):
            raise InvalidCorrection("cells must be a list of cell ids")
        if all(isinstance(item, Mapping) for item in cells):
            return dict(_as_pairs(cells, name="cells"))
        if not all(isinstance(cell, str) for cell in cells):
            raise InvalidCorrection("cells must be a list of cell ids")
        return cycle_permutation(list(cells))

    old_cell = _first(body, ("from_cell", "old_cell", "source_cell"))
    new_cell = _first(body, ("to_cell", "new_cell", "target_cell"))
    if old_cell is not None or new_cell is not None:
        if old_cell is None or new_cell is None:
            raise InvalidCorrection("name both the old and the new cell id")
        return {str(old_cell): str(new_cell), str(new_cell): str(old_cell)}

    # A bare body of {"old": "new"} pairs is accepted as an explicit map when
    # no other field carries cell ids.
    bare = {
        str(key): str(value)
        for key, value in body.items()
        if key not in _METADATA_KEYS and isinstance(value, str)
    }
    if bare:
        return bare

    raise InvalidCorrection(
        "a correction must name cell ids to permute: provide permutation, "
        "swaps or an ordered cells list"
    )


async def _correction_request(
    request: Request,
    period: Optional[str],
    archive: Archive,
) -> JSONResponse:
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="request body must be JSON") from exc
    if not isinstance(body, Mapping):
        raise HTTPException(status_code=422, detail="request body must be an object")

    body_period = _first(body, _PERIOD_KEYS)
    query_period = _first(request.query_params, _PERIOD_KEYS)
    period = (
        period
        or (str(body_period) if body_period is not None else None)
        or (str(query_period) if query_period is not None else None)
    )
    if not period or not str(period).strip():
        raise HTTPException(
            status_code=422, detail="statistical period must not be blank"
        )
    seen = _first(body, _SEEN_BATCH_KEYS)
    if seen is None:
        seen = _first(request.query_params, _SEEN_BATCH_KEYS)
    if not isinstance(seen, str) or not seen:
        raise HTTPException(
            status_code=422,
            detail=(
                "seen_batch_id is required and must be the batch id currently "
                "reported by GET /archive/{period}"
            ),
        )

    try:
        permutation = validate_permutation(_correction_permutation(body))
    except InvalidCorrection as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        result = archive.correct_batch(period, seen, permutation)
    except UnknownPeriod as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BatchChanged as exc:
        return JSONResponse(
            status_code=409,
            content=CorrectionFailureResponse(
                detail=(
                    "the archive changed after the seen batch; correction "
                    "refused and the current batch is unchanged"
                ),
                period=period,
                seen_batch_id=exc.expected,
                current_batch_id=exc.current,
            ).model_dump(),
        )
    except CorrectionProblem as exc:
        return JSONResponse(
            status_code=422,
            content=CorrectionFailureResponse(
                detail=str(exc),
                period=period,
                seen_batch_id=seen,
                current_batch_id=seen,
                upper_cells=list(exc.upper_cells),
                missing_cells=list(exc.missing_cells),
                negative_cells=list(exc.negative_cells),
                missing_regions=list(exc.missing_regions),
            ).model_dump(),
        )
    except InvalidCorrection as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InvalidBatch as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return JSONResponse(
        status_code=201,
        content={
            "period": result.period,
            "batch_id": result.batch_id,
            "new_batch_id": result.batch_id,
            "batch_no": result.batch_id,
            "previous_batch_id": result.seen_batch_id,
            "old_batch_id": result.seen_batch_id,
            "seen_batch_id": result.seen_batch_id,
            "created_at": result.created_at,
            "cell_count": result.cell_count,
            "total_population": result.total_population,
            "total": result.total_population,
            "permutation": dict(result.permutation),
            "corrections": [
                {
                    "source": source,
                    "target": target,
                    "old": source,
                    "new": target,
                }
                for source, target in result.permutation
            ],
            "moves": [
                {
                    "source": source,
                    "target": target,
                    "old_cell": source,
                    "new_cell": target,
                    "from_cell": source,
                    "to_cell": target,
                    "cell": source,
                    "population": population,
                }
                for source, target, population in result.moves
            ],
            "recomputed": [
                {
                    "cell": cell,
                    "previous_population": previous,
                    "population": population,
                    "old_population": previous,
                    "new_population": population,
                }
                for cell, previous, population in result.recomputed
            ],
            "upper_cells": [
                cell for cell, _, _ in result.recomputed
            ],
        },
    )


@app.post("/archive/{period}/correct")
async def correct_cells_in_period(
    period: str,
    request: Request,
    archive: Archive = Depends(get_archive),
) -> JSONResponse:
    return await _correction_request(request, period, archive)


@app.post("/archive/{period}/correction")
async def correct_cells_in_period_alias(
    period: str,
    request: Request,
    archive: Archive = Depends(get_archive),
) -> JSONResponse:
    return await _correction_request(request, period, archive)


@app.post("/archive/{period}/corrections")
async def correct_cells_in_period_plural(
    period: str,
    request: Request,
    archive: Archive = Depends(get_archive),
) -> JSONResponse:
    return await _correction_request(request, period, archive)


@app.post("/archive/{period}/swap")
async def swap_cells_in_period(
    period: str,
    request: Request,
    archive: Archive = Depends(get_archive),
) -> JSONResponse:
    return await _correction_request(request, period, archive)


@app.post("/archive/{period}/cells/correct")
async def correct_cells_nested(
    period: str,
    request: Request,
    archive: Archive = Depends(get_archive),
) -> JSONResponse:
    return await _correction_request(request, period, archive)


@app.post("/correction")
async def correct_cells(
    request: Request,
    archive: Archive = Depends(get_archive),
) -> JSONResponse:
    return await _correction_request(request, None, archive)


@app.post("/census/correct")
async def correct_cells_census_alias(
    request: Request,
    archive: Archive = Depends(get_archive),
) -> JSONResponse:
    return await _correction_request(request, None, archive)


@app.post("/census/correction")
async def correct_cells_census_correction_alias(
    request: Request,
    archive: Archive = Depends(get_archive),
) -> JSONResponse:
    return await _correction_request(request, None, archive)


@app.put("/archive/{period}/correct")
async def correct_cells_in_period_put(
    period: str,
    request: Request,
    archive: Archive = Depends(get_archive),
) -> JSONResponse:
    return await _correction_request(request, period, archive)


@app.put("/archive/{period}/correction")
async def correct_cells_in_period_put_alias(
    period: str,
    request: Request,
    archive: Archive = Depends(get_archive),
) -> JSONResponse:
    return await _correction_request(request, period, archive)


@app.put("/correction")
async def correct_cells_put(
    request: Request,
    archive: Archive = Depends(get_archive),
) -> JSONResponse:
    return await _correction_request(request, None, archive)


@app.post("/census/query", response_model=CensusResponse)
def query_census(
    request: QueryRequest,
    archive: Archive = Depends(get_archive),
) -> CensusResponse:
    with archive.snapshot(request.period) as snapshot:
        # Phase 1: derive the minimal expression from the catalog alone.  No
        # headcount is read and no finest cell is enumerated.
        try:
            analysis = analyze(snapshot.catalog(), request.cells, depth=MAX_DEPTH)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not analysis.feasible:
            return _missing_response(request.period, analysis)
        solution = analysis.solution

        # Phase 2: read only the cells that occur in the expression.
        populations = snapshot.populations(term.cell for term in solution.terms)
        total = solution.evaluate(populations.__getitem__)

    return CensusResponse(
        period=request.period,
        cells=list(dict.fromkeys(request.cells)),
        expression=solution.render(),
        terms=[
            ExpressionTermModel(cell=term.cell, coefficient=term.coefficient)
            for term in solution.terms
        ],
        retrieved_cells=solution.retrieved_cells,
        total_population=total,
    )


def _missing_response(period: str, analysis) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=MissingArchiveResponse(
            detail=(
                "the archive for this period cannot support an exact "
                "expression for the requested cells; archive the listed "
                "missing blocks and retry"
            ),
            period=period,
            missing_blocks=list(analysis.missing_blocks),
            truncated=analysis.missing_truncated,
        ).model_dump(),
    )
