"""Unit tests for the _sum_usage state reducer (yaml_graph.py)."""
from __future__ import annotations


def test_sum_usage_none_a_returns_b():
    from app.infrastructure.orchestration.yaml_graph import _sum_usage

    usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    assert _sum_usage(None, usage) == usage


def test_sum_usage_none_b_returns_a():
    from app.infrastructure.orchestration.yaml_graph import _sum_usage

    usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    assert _sum_usage(usage, None) == usage


def test_sum_usage_both_missing_returns_falsy():
    from app.infrastructure.orchestration.yaml_graph import _sum_usage

    assert not _sum_usage(None, None)
    assert not _sum_usage({}, None)
    assert not _sum_usage(None, {})


def test_sum_usage_sums_matching_fields():
    from app.infrastructure.orchestration.yaml_graph import _sum_usage

    a = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    b = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

    assert _sum_usage(a, b) == {"input_tokens": 110, "output_tokens": 55, "total_tokens": 165}


def test_sum_usage_tolerates_missing_keys_on_either_side():
    from app.infrastructure.orchestration.yaml_graph import _sum_usage

    a = {"input_tokens": 100}
    b = {"output_tokens": 5, "extra_field": 3}

    assert _sum_usage(a, b) == {"input_tokens": 100, "output_tokens": 5, "extra_field": 3}


def test_sum_usage_sums_extra_numeric_keys():
    from app.infrastructure.orchestration.yaml_graph import _sum_usage

    a = {"input_tokens": 10, "cache_read_tokens": 4}
    b = {"input_tokens": 5, "cache_read_tokens": 6}

    assert _sum_usage(a, b) == {"input_tokens": 15, "cache_read_tokens": 10}


def test_sum_usage_never_raises_on_non_dict_a():
    from app.infrastructure.orchestration.yaml_graph import _sum_usage

    b = {"input_tokens": 5}
    assert _sum_usage("legacy string", b) == b


def test_sum_usage_never_raises_on_non_dict_b():
    from app.infrastructure.orchestration.yaml_graph import _sum_usage

    a = {"input_tokens": 5}
    assert _sum_usage(a, "legacy string") == a


def test_sum_usage_both_non_dict_returns_empty_dict():
    from app.infrastructure.orchestration.yaml_graph import _sum_usage

    assert _sum_usage("foo", "bar") == {}


def test_sum_usage_legacy_string_value_in_dict_coerced_to_zero():
    from app.infrastructure.orchestration.yaml_graph import _sum_usage

    a = {"input_tokens": "not-a-number", "output_tokens": 10}
    b = {"input_tokens": 5, "output_tokens": 3}

    assert _sum_usage(a, b) == {"input_tokens": 5, "output_tokens": 13}


def test_sum_usage_non_numeric_value_does_not_raise():
    from app.infrastructure.orchestration.yaml_graph import _sum_usage

    a = {"input_tokens": None}
    b = {"input_tokens": object()}

    assert _sum_usage(a, b) == {"input_tokens": 0}


def test_sum_usage_keeps_ints_as_ints():
    from app.infrastructure.orchestration.yaml_graph import _sum_usage

    a = {"input_tokens": 10}
    b = {"input_tokens": 5}

    result = _sum_usage(a, b)
    assert result == {"input_tokens": 15}
    assert isinstance(result["input_tokens"], int)
