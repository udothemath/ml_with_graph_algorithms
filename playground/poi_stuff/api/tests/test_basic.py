"""Tests for basic.py."""
from basic import add


def test_add():
    """Test for function add()."""
    a = 6
    b = 7
    expected_result = 13
    assert expected_result == add(a, b)
