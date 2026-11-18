"""Every comparison recorded in `reference_checks.py` as a test."""

import pytest

from reference_checks import RESULTS


@pytest.mark.parametrize("section, name, reference, value, expected, ok", RESULTS,
                         ids=[r[1] for r in RESULTS])
def test_reference_value(section, name, reference, value, expected, ok):
    assert ok, f"{name}: computed {value!r}, expected {expected!r} ({reference})"
