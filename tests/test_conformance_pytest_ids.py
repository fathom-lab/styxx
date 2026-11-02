"""The conformance recorder keeps pytest's parametrized node ids stable across pytest versions.

pytest 9.2 names a parametrized case with a string value over 100 characters by argument name and index
where earlier pytest wrote the escaped value. Those node ids are the vectors' `sources`, inside set_sha256,
so without this the committed set "drifted" under a new pytest while every vector still replayed.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conformance.sworn import gen_vectors  # noqa: E402

LEGACY = ["-o", "parametrize_long_str_id_strategy=legacy"]


@pytest.mark.parametrize("version", ["9.2.0", "9.2.0rc1", "9.10.1", "10.0", "11"])
def test_pytest_9_2_and_later_get_the_legacy_names(version):
    assert gen_vectors.pytest_id_args(version) == LEGACY


@pytest.mark.parametrize("version", ["9.1.1", "9.0.3", "8.4.2", "7.4.4", "garbage"])
def test_earlier_pytest_is_left_as_it_was(version):
    assert gen_vectors.pytest_id_args(version) == []


def test_the_installed_pytest_is_read_when_no_version_is_given():
    expected = LEGACY if tuple(int(x) for x in pytest.__version__.split(".")[:2] if x.isdigit()) >= (9, 2) else []
    assert gen_vectors.pytest_id_args() == expected


def test_the_recorder_passes_the_arguments_to_its_pytest_child():
    src = inspect.getsource(gen_vectors.run_recorder)
    assert "*pytest_id_args()" in src
