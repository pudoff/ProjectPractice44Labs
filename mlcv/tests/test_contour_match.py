"""Contour quality: one detected contour - one reference contour."""
import sys
from pathlib import Path

import pytest
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

match_contours = pytest.importorskip("contour_match", reason="mlcv/eval is not in this image").match_contours


def test_extra_copies_lower_precision():
    ref = [box(0, 0, 10, 10)]
    r = match_contours(ref, [box(0, 0, 10, 10), box(0, 0, 10, 10), box(0, 0, 10, 10)])
    assert r["found"] == 1 and r["extra"] == 2 and r["copies"] == 2
    assert r["precision"] == pytest.approx(1 / 3, abs=0.001) and r["recall"] == 1.0


def test_one_detected_contour_cannot_close_several_references():
    """Two reference contours that are nearly the same shape: the old metric counted both as found."""
    ref = [box(0, 0, 10, 10), box(0, 0, 10, 9.8)]
    r = match_contours(ref, [box(0, 0, 10, 10)])
    assert r["found_many_to_one"] == 2
    assert r["found"] == 1 and r["missed"] == 1 and r["recall"] == 0.5 and r["precision"] == 1.0


def test_assignment_finds_the_most_pairs():
    """A greedy best-IoU choice would pair d0 with r1 and leave r0 alone."""
    r0, r1 = box(0, 0, 10, 10), box(0, 0, 10, 9.5)
    d0, d1 = box(0, 0, 10, 9.6), box(0, 0, 10, 9.3)
    r = match_contours([r0, r1], [d0, d1])
    assert r["found"] == 2 and r["extra"] == 0 and r["missed"] == 0


def test_low_iou_is_not_a_match_and_empty_inputs_work():
    r = match_contours([box(0, 0, 10, 10)], [box(0, 0, 10, 5)])
    assert r["found"] == 0 and r["missed"] == 1 and r["extra"] == 1 and r["copies"] == 0
    assert match_contours([], [])["found"] == 0
    assert match_contours([box(0, 0, 1, 1)], [])["recall"] == 0.0
    assert match_contours([], [box(0, 0, 1, 1)])["precision"] == 0.0
