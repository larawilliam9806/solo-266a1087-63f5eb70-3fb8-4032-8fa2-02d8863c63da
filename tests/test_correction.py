import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.correction import (
    CorrectionProblem,
    InvalidCorrection,
    Recompute,
    cycle_permutation,
    plan_correction,
    validate_permutation,
)


def plan(entries, permutation):
    return plan_correction(dict(entries), permutation)


def as_dict(result):
    return dict(result.cells)


# ---------------------------------------------------------------- validation


def test_permutation_requires_two_to_eight_same_level_cells():
    assert cycle_permutation(["01", "02"]) == {"01": "02", "02": "01"}
    with pytest.raises(InvalidCorrection):
        validate_permutation({"01": "01"})
    with pytest.raises(InvalidCorrection):
        validate_permutation({f"0{i}": f"0{i}" for i in range(9)})
    with pytest.raises(InvalidCorrection):
        # Mixed levels: "1" and "2" are shallower than "01"/"02".
        validate_permutation({"01": "02", "02": "01", "1": "2", "2": "1"})


def test_permutation_must_stay_within_its_own_positions():
    with pytest.raises(InvalidCorrection):
        validate_permutation({"01": "02", "03": "04"})
    with pytest.raises(InvalidCorrection):
        validate_permutation({"01": "02", "02": "01", "010": "020"})


def test_permutation_rejects_bad_cells():
    with pytest.raises(InvalidCorrection):
        validate_permutation({"4": "1", "1": "4"})
    with pytest.raises(InvalidCorrection):
        validate_permutation({"1" * 13: "0", "0": "1" * 13})


# ---------------------------------------------------------------- basic swap


def test_swap_moves_suffixes_and_counts():
    result = plan(
        {
            "0": 100,
            "01": 10,
            "010": 4,
            "0102": 3,
            "02": 20,
            "03": 30,
            "1": 200,
        },
        {"01": "02", "02": "01"},
    )
    cells = as_dict(result)
    # Suffixes are preserved, counts travel with their records.
    assert cells["020"] == 4
    assert cells["0202"] == 3
    assert cells["01"] == 20
    assert cells["02"] == 10
    assert cells["03"] == 30
    assert cells["1"] == 200
    # "0" is a strict ancestor of both positions and its source set is
    # unchanged, so it is not recomputed.
    assert cells["0"] == 100
    assert result.recomputed == ()
    assert {(m.source, m.target, m.population) for m in result.moves} == {
        ("01", "02", 10),
        ("010", "020", 4),
        ("0102", "0202", 3),
        ("02", "01", 20),
    }


def test_swap_keeps_cells_outside_positions_untouched():
    result = plan({"0": 10, "1": 20, "2": 30, "01": 1, "02": 2}, {"01": "02", "02": "01"})
    cells = as_dict(result)
    assert cells["0"] == 10 and cells["1"] == 20 and cells["2"] == 30


def test_cycle_of_three_positions():
    result = plan(
        {"1": 60, "10": 1, "11": 2, "12": 3},
        cycle_permutation(["10", "11", "12"]),
    )
    cells = as_dict(result)
    assert cells == {"1": 60, "10": 3, "11": 1, "12": 2}


def test_deepest_level_swap():
    deep_a, deep_b = "012301230123", "321032103210"
    result = plan({deep_a: 5, deep_b: 7}, {deep_a: deep_b, deep_b: deep_a})
    assert as_dict(result) == {deep_a: 7, deep_b: 5}


# ------------------------------------------------- upper-level recomputation


def test_cross_parent_swap_recomputes_both_parents():
    result = plan(
        {
            "": 100,
            "0": 40,
            "00": 10,
            "1": 60,
            "10": 15,
        },
        {"00": "10", "10": "00"},
    )
    cells = as_dict(result)
    # "0" loses region 00 (10) and gains region 10 (15).
    assert cells["0"] == 45
    # "1" loses 10 and gains 00.
    assert cells["1"] == 55
    # Root sources are unchanged: it still covers the same regions below.
    assert cells[""] == 100
    assert {(r.cell, r.previous_population, r.population) for r in result.recomputed} == {
        ("0", 40, 45),
        ("1", 60, 55),
    }


def test_unknown_region_is_derived_from_precorrection_archive():
    # Districts are not archived individually; "0" is total 30 and its four
    # children are known, so each region value is expressible.
    result = plan(
        {
            "0": 30,
            "00": 5,
            "01": 6,
            "02": 7,
            "03": 12,
            "1": 50,
            "10": 9,
        },
        {"00": "10", "10": "00"},
    )
    cells = as_dict(result)
    assert cells["0"] == 30 - 5 + 9
    assert cells["1"] == 50 - 9 + 5


def test_deriving_one_region_without_individual_counts():
    # Only the parent total and a sibling breakdown are stored.  Region 02 is
    # not stored; the solver derives it exactly, without enumerating leaves.
    result = plan(
        {
            "0": 100,
            "00": 10,
            "01": 20,
            "02": 30,
            "03": 40,
            "1": 200,
            "10": 50,
        },
        {"02": "10", "10": "02"},
    )
    cells = as_dict(result)
    assert cells["0"] == 100 - 30 + 50
    assert cells["1"] == 200 - 50 + 30


def test_sibling_swap_works_without_individual_counts():
    # "0" holds only its total; the districts below it are not isolated.
    # Swapping siblings leaves "0"'s source set unchanged, so it can proceed.
    result = plan({"0": 100}, {"00": "01", "01": "00"})
    assert as_dict(result) == {"0": 100}
    assert result.moves == ()
    assert result.recomputed == ()


def test_sibling_swap_moves_children_even_without_parent_total():
    result = plan({"00": 4, "01": 5}, {"00": "01", "01": "00"})
    assert as_dict(result) == {"00": 5, "01": 4}


def test_swap_between_branches_under_root_recomputes_root():
    result = plan({"": 10, "0": 4, "1": 6}, {"0": "1", "1": "0"})
    cells = as_dict(result)
    assert cells["0"] == 6 and cells["1"] == 4
    assert cells[""] == 10  # same two regions, no change


# ---------------------------------------------------------------- failures


def test_missing_upper_cell_is_reported():
    # To move region 02 into "1", the pre-correction archive must know region
    # 02; "0" total alone cannot isolate it (its children are unknown).
    with pytest.raises(CorrectionProblem) as info:
        plan({"0": 100, "1": 50, "10": 5}, {"02": "10", "10": "02"})
    problem = info.value
    assert problem.missing_cells == ("0", "1")
    assert problem.upper_cells == ("0", "1")
    assert problem.missing_regions


def test_negative_recomputation_names_upper_cell():
    # "0" stores 3 while the archive also says region 00 alone is 5; moving
    # 00 out drives "0" negative.
    with pytest.raises(CorrectionProblem) as info:
        plan({"0": 3, "00": 5, "1": 20, "10": 0}, {"00": "10", "10": "00"})
    problem = info.value
    assert "0" in problem.negative_cells
    assert "10" in problem.negative_cells or "0" in problem.upper_cells
    assert problem.upper_cells == ("0",)


def test_problem_reports_only_affected_upper_cells():
    result = plan(
        {
            "0": 100,
            "00": 10,
            "01": 90,
            "1": 5,
            "10": 4,
            "2": 8,
        },
        {"00": "10", "10": "00"},
    )
    cells = as_dict(result)
    # "0" becomes 100 - 10 + 4 = 94, "1" becomes 5 - 4 + 10 = 11: all exact.
    assert cells["0"] == 94
    assert cells["1"] == 11
    assert cells["2"] == 8
    assert result.recomputed == (
        Recompute("0", 100, 94),
        Recompute("1", 5, 11),
    )


def test_recompute_of_deeper_ancestor_uses_exact_region():
    result = plan(
        {
            "0": 60,
            "00": 20,
            "000": 8,
            "001": 12,
            "1": 40,
            "100": 15,
            "101": 20,
        },
        {"000": "100", "100": "000"},
    )
    cells = as_dict(result)
    assert cells["00"] == 20 - 8 + 15
    # "0" loses region 000 (8) and gains region 100 (15); "1" the reverse.
    assert cells["0"] == 60 - 8 + 15
    assert cells["1"] == 40 - 15 + 8


def test_correction_preserves_total_population():
    result = plan(
        {"": 1000, "0": 400, "00": 100, "01": 300, "1": 600, "10": 250},
        {"00": "10", "10": "00"},
    )
    before = sum({"": 1000, "0": 400, "00": 100, "01": 300, "1": 600, "10": 250}.values())
    assert sum(pop for _, pop in result.cells) == before


def test_cross_branch_cycle_recomputes_every_affected_ancestor():
    # A three-cycle spanning three top-level branches: every branch's upper
    # record must lose its outgoing region and gain its incoming one.
    entries = {
        "": 300,
        "0": 100,
        "00": 40,
        "01": 60,
        "1": 100,
        "10": 55,
        "11": 45,
        "2": 100,
        "20": 70,
        "21": 30,
    }
    result = plan(entries, cycle_permutation(["00", "10", "20"]))
    cells = as_dict(result)
    # 00 (40) -> 10, 10 (55) -> 20, 20 (70) -> 00
    assert cells["0"] == 100 - 40 + 70
    assert cells["1"] == 100 - 55 + 40
    assert cells["2"] == 100 - 70 + 55
    assert cells[""] == 300
    assert sum(cells.values()) == sum(entries.values())


def test_cross_branch_cycle_reports_negative_for_each_offending_upper_cell():
    entries = {
        "0": 30,
        "00": 40,  # archive already inconsistent: child exceeds parent
        "1": 10,
        "10": 0,
    }
    with pytest.raises(CorrectionProblem) as info:
        plan(entries, {"00": "10", "10": "00"})
    assert "0" in info.value.negative_cells


def test_identity_permutation_is_rejected():
    with pytest.raises(InvalidCorrection):
        validate_permutation({"01": "01", "02": "02"})


def test_partial_identity_moves_only_changed_positions():
    result = plan(
        {"0": 10, "01": 1, "02": 2, "03": 3},
        {"01": "02", "02": "01", "03": "03"},
    )
    cells = as_dict(result)
    assert cells["01"] == 2 and cells["02"] == 1 and cells["03"] == 3
    assert {m.source for m in result.moves} == {"01", "02"}
