import sqlite3
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.archive import Archive, BatchChanged
from app.main import app, get_archive


@pytest.fixture()
def archive(tmp_path):
    return Archive(tmp_path / "archive.sqlite3")


@pytest.fixture()
def client(archive):
    app.dependency_overrides[get_archive] = lambda: archive
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def submit(client, period, cells, replace=False):
    items = cells.items() if isinstance(cells, dict) else cells
    return client.post(
        "/archive",
        json={
            "period": period,
            "replace": replace,
            "cells": [{"cell": c, "population": p} for c, p in items],
        },
    )


def query(client, period, cells):
    return client.post("/census/query", json={"period": period, "cells": cells})


def batch_id(client, period="p"):
    return client.get(f"/archive/{period}").json()["batch_id"]


def correct(client, period, seen, permutation, **extra):
    body = {"period": period, "seen_batch_id": seen, "permutation": permutation}
    body.update(extra)
    return client.post(f"/archive/{period}/correct", json=body)


# --------------------------------------------------------------- batch ids


def test_batch_id_is_reported_and_changes_on_replace(client):
    first = submit(client, "p", [("0", 10), ("1", 20)]).json()["batch_id"]
    assert first
    assert batch_id(client, "p") == first
    # A failed (non-replace) submission does not mint a new batch id.
    assert submit(client, "p", [("0", 1)]).status_code == 409
    assert batch_id(client, "p") == first
    second = submit(client, "p", [("0", 30)], replace=True).json()["batch_id"]
    assert second != first


def test_correction_mints_new_batch_id_and_reads_switch(client):
    r = submit(client, "p", [("0", 100), ("01", 10), ("02", 20), ("1", 50), ("10", 5)])
    seen = r.json()["batch_id"]
    moved = correct(client, "p", seen, {"01": "02", "02": "01"})
    assert moved.status_code == 201, moved.text
    body = moved.json()
    assert body["previous_batch_id"] == seen
    assert body["batch_id"] != seen
    assert batch_id(client, "p") == body["batch_id"]
    # The query endpoint reads the new batch immediately.
    assert query(client, "p", ["01"]).json()["total_population"] == 20
    assert query(client, "p", ["02"]).json()["total_population"] == 10


def test_correction_reports_moves_and_recomputed(client):
    submit(client, "p", {"0": 40, "00": 10, "1": 60, "10": 15})
    seen = batch_id(client, "p")
    body = correct(client, "p", seen, {"00": "10", "10": "00"}).json()
    assert body["permutation"] == {"00": "10", "10": "00"}
    assert {(m["source"], m["target"], m["population"]) for m in body["moves"]} == {
        ("00", "10", 10),
        ("10", "00", 15),
    }
    assert {(r["cell"], r["population"]) for r in body["recomputed"]} == {
        ("0", 45),
        ("1", 55),
    }


# ----------------------------------------------------------- seen-batch guard


def test_stale_seen_batch_is_refused_and_current_batch_kept(client):
    seen = submit(client, "p", [("0", 10), ("1", 20)]).json()["batch_id"]
    replaced = submit(client, "p", [("0", 99), ("1", 1)], replace=True).json()["batch_id"]
    r = correct(client, "p", seen, {"0": "1", "1": "0"})
    assert r.status_code == 409, r.text
    body = r.json()
    assert body["seen_batch_id"] == seen
    assert body["current_batch_id"] == replaced
    # The current batch is untouched.
    assert query(client, "p", ["0"]).json()["total_population"] == 99
    assert batch_id(client, "p") == replaced


def test_second_correction_with_old_batch_id_is_refused(client):
    seen = submit(client, "p", [("0", 10), ("1", 20)]).json()["batch_id"]
    assert correct(client, "p", seen, {"0": "1", "1": "0"}).status_code == 201
    again = correct(client, "p", seen, {"0": "1", "1": "0"})
    assert again.status_code == 409


def test_conflict_inside_write_transaction_is_detected(archive, monkeypatch):
    """A replacement between planning and writing must abort the correction."""
    archive.replace_batch("p", [("0", 10), ("1", 20)])
    seen = archive.batch_info("p").batch_id
    real_connect = Archive._connect
    state = {"injected": False}

    def connect_then_replace(self, *, write=False):
        # Before the correction's write transaction opens, replace the batch
        # through a different connection; the batch-id re-check must fire.
        if write and not state["injected"]:
            state["injected"] = True
            archive.replace_batch("p", [("0", 77), ("1", 88)], allow_replace=True)
        return real_connect(self, write=write)

    monkeypatch.setattr(Archive, "_connect", connect_then_replace)
    with pytest.raises(BatchChanged):
        archive.correct_batch("p", seen, {"0": "1", "1": "0"})
    monkeypatch.setattr(Archive, "_connect", real_connect)
    assert archive.batch_info("p").total_population == 165
    assert query_values(archive, "p", ["0"]) == 77


def query_values(archive, period, cells):
    with archive.snapshot(period) as snapshot:
        return snapshot.populations(cells)[cells[0]]


# ------------------------------------------------------------- request forms


def test_correction_accepts_cells_list_as_cycle(client):
    submit(client, "p", {"0": 60, "10": 1, "11": 2, "12": 3})
    seen = batch_id(client, "p")
    r = client.post(
        "/archive/p/correct",
        json={"seen_batch_id": seen, "cells": ["10", "11", "12"]},
    )
    assert r.status_code == 201, r.text
    assert query(client, "p", ["10"]).json()["total_population"] == 3
    assert query(client, "p", ["11"]).json()["total_population"] == 1
    assert query(client, "p", ["12"]).json()["total_population"] == 2


def test_correction_accepts_swaps_and_route_without_period(client):
    submit(client, "p", {"0": 60, "10": 1, "11": 2, "1": 40, "12": 5, "13": 6})
    seen = batch_id(client, "p")
    r = client.post(
        "/correction",
        json={
            "period": "p",
            "seen_batch": seen,
            "swaps": [["10", "11"], ["12", "13"]],
        },
    )
    assert r.status_code == 201, r.text
    assert query(client, "p", ["10"]).json()["total_population"] == 2
    assert query(client, "p", ["13"]).json()["total_population"] == 5


def test_correction_accepts_from_to_lists(client):
    submit(client, "p", {"0": 60, "10": 1, "11": 2})
    seen = batch_id(client, "p")
    r = client.post(
        "/archive/p/correction",
        json={"seen_batch_id": seen, "from_cells": ["10", "11"], "to_cells": ["11", "10"]},
    )
    assert r.status_code == 201, r.text
    assert query(client, "p", ["10"]).json()["total_population"] == 2


def test_correction_accepts_bare_map_body(client):
    submit(client, "p", {"0": 60, "10": 1, "11": 2})
    seen = batch_id(client, "p")
    r = client.post("/correction", json={"period": "p", "seen": seen, "10": "11", "11": "10"})
    assert r.status_code == 201, r.text
    assert query(client, "p", ["10"]).json()["total_population"] == 2


def test_correction_accepts_structured_swaps(client):
    submit(client, "p", {"0": 60, "10": 1, "11": 2})
    seen = batch_id(client, "p")
    r = client.post(
        "/correction",
        json={
            "period": "p",
            "batch_no": seen,
            "cell_corrections": [
                {"old_cell": "10", "new_cell": "11"},
                {"old_cell": "11", "new_cell": "10"},
            ],
        },
    )
    assert r.status_code == 201, r.text
    assert query(client, "p", ["10"]).json()["total_population"] == 2


def test_identity_correction_is_rejected(client):
    submit(client, "p", {"0": 1, "1": 2})
    seen = batch_id(client, "p")
    r = correct(client, "p", seen, {"0": "0", "1": "1"})
    assert r.status_code == 422
    assert batch_id(client, "p") == seen


# ---------------------------------------------------------------- rejections


def test_missing_seen_batch_id_is_rejected(client):
    submit(client, "p", {"0": 1, "1": 2})
    r = client.post("/archive/p/correct", json={"cells": ["0", "1"]})
    assert r.status_code == 422


def test_bad_permutation_is_rejected_and_batch_kept(client):
    submit(client, "p", {"0": 1, "1": 2})
    seen = batch_id(client, "p")
    for permutation in (
        {"0": "0"},
        {"0": "1", "1": "2"},  # not one-to-one onto its own positions
        {"0": "1", "1": "0", "2": "3", "3": "2", "4": "5", "5": "4", "6": "7", "7": "6", "8": "9", "9": "8"},
        {"0": "2", "1": "0"},  # mixed target set
    ):
        r = correct(client, "p", seen, permutation)
        assert r.status_code == 422, (permutation, r.text)
    assert batch_id(client, "p") == seen
    assert query(client, "p", ["0"]).json()["total_population"] == 1


def test_missing_archive_is_reported_per_upper_cell_and_batch_kept(client):
    # To isolate region 02 the archive would need 02 or its children; only the
    # total of "0" is stored, so "0" is the missing upper cell.
    submit(client, "p", {"0": 100, "1": 50, "10": 5})
    seen = batch_id(client, "p")
    r = correct(client, "p", seen, {"02": "10", "10": "02"})
    assert r.status_code == 422, r.text
    body = r.json()
    assert set(body["missing_cells"]) == {"0", "1"}
    assert set(body["upper_cells"]) == {"0", "1"}
    assert body["missing_regions"]
    assert batch_id(client, "p") == seen
    assert query(client, "p", ["0"]).json()["total_population"] == 100


def test_unrelated_upper_cells_are_not_reported(client):
    # "2" is a bystander branch: the correction touches no region under it, so
    # even though its totals cannot help isolate anything it must not be named.
    submit(
        client,
        "p",
        {
            "0": 100,
            "00": 30,
            "01": 70,
            "1": 50,
            "2": 999,
            "3": 5,
        },
    )
    seen = batch_id(client, "p")
    # Region 10 is missing, so "0" and "1" are reported; "2" and "3" are not.
    r = correct(client, "p", seen, {"00": "10", "10": "00"})
    assert r.status_code == 422, r.text
    assert set(r.json()["upper_cells"]) == {"0", "1"}


def test_query_reflects_correction_expression_and_totals(client):
    submit(
        client,
        "p",
        {"0": 100, "00": 15, "01": 85, "1": 50, "10": 20},
    )
    before = query(client, "p", ["00"]).json()
    assert before["total_population"] == 15
    assert before["expression"] == 'pop("00")'
    seen = batch_id(client, "p")
    assert correct(client, "p", seen, {"00": "10", "10": "00"}).status_code == 201
    after = query(client, "p", ["00"]).json()
    assert after["total_population"] == 20
    # "01" is untouched by the swap.
    assert query(client, "p", ["01"]).json()["total_population"] == 85
    meta = client.get("/archive/p").json()
    assert meta["cell_count"] == 5
    # Batch metadata sums the stored rows (aggregates and their children), so
    # the swap must not change it: 100 + 15 + 85 + 50 + 20.
    assert meta["total_population"] == 270


def test_negative_recomputation_is_reported_and_batch_kept(client):
    submit(client, "p", {"0": 3, "00": 5, "1": 20, "10": 0})
    seen = batch_id(client, "p")
    r = correct(client, "p", seen, {"00": "10", "10": "00"})
    assert r.status_code == 422, r.text
    body = r.json()
    assert "0" in body["negative_cells"]
    assert batch_id(client, "p") == seen
    assert query(client, "p", ["0"]).json()["total_population"] == 3


def test_unknown_period_correction_is_404(client):
    r = correct(client, "nope", "whatever", {"0": "1", "1": "0"})
    assert r.status_code == 404


def test_correction_needs_valid_json_object(client):
    r = client.post(
        "/archive/p/correct",
        content=b"[]",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 422


# ------------------------------------------------------- old database upgrade


def make_legacy_database(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE batches (
            period     TEXT PRIMARY KEY,
            created_at TEXT NOT NULL
        );
        CREATE TABLE cells (
            period     TEXT    NOT NULL,
            cell       TEXT    NOT NULL,
            population INTEGER NOT NULL,
            PRIMARY KEY (period, cell),
            FOREIGN KEY (period) REFERENCES batches(period) ON DELETE CASCADE
        );
        """
    )
    conn.execute(
        "INSERT INTO batches(period, created_at) VALUES('old', '2020-01-01T00:00:00+00:00')"
    )
    conn.executemany(
        "INSERT INTO cells(period, cell, population) VALUES('old', ?, ?)",
        [("0", 70), ("1", 30)],
    )
    conn.commit()
    conn.close()


def test_legacy_database_is_upgraded_and_usable(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    make_legacy_database(path)
    archive = Archive(path)
    info = archive.batch_info("old")
    assert info.batch_id
    assert info.total_population == 100

    app.dependency_overrides[get_archive] = lambda: archive
    try:
        with TestClient(app) as client:
            assert query(client, "old", ["0"]).json()["total_population"] == 70
            seen = batch_id(client, "old")
            r = correct(client, "old", seen, {"0": "1", "1": "0"})
            assert r.status_code == 201, r.text
            assert query(client, "old", ["0"]).json()["total_population"] == 30
    finally:
        app.dependency_overrides.clear()


def test_two_legacy_batches_get_distinct_ids(tmp_path):
    path = tmp_path / "legacy2.sqlite3"
    make_legacy_database(path)
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO batches(period, created_at) VALUES('old2', '2020-01-01T00:00:00+00:00')"
    )
    conn.execute("INSERT INTO cells(period, cell, population) VALUES('old2', '2', 5)")
    conn.commit()
    conn.close()
    archive = Archive(path)
    assert archive.batch_info("old").batch_id != archive.batch_info("old2").batch_id
