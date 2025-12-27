from __future__ import annotations

from collections.abc import Iterable


def _assert_iterable_nonempty(x: object) -> None:
    assert isinstance(x, Iterable)
    assert len(list(x)) > 0


def test_constants_module_exports_expected_names() -> None:
    """
    Validate that eb_features.panel.constants exposes the expected public constants.

    If you rename constants in the library, update this test to match the new API.
    """
    import eb_features.panel.constants as c

    expected_names = {
        # defaults
        "DEFAULT_LAG_STEPS",
        "DEFAULT_ROLLING_WINDOWS",
        "DEFAULT_ROLLING_STATS",
        "DEFAULT_CALENDAR_FEATURES",
        # allowed keys
        "ALLOWED_ROLLING_STATS",
        "ALLOWED_CALENDAR_FEATURES",
    }

    for name in expected_names:
        assert hasattr(c, name), f"Missing constant: {name}"


def test_default_constants_are_nonempty_iterables() -> None:
    import eb_features.panel.constants as c

    _assert_iterable_nonempty(c.DEFAULT_LAG_STEPS)
    _assert_iterable_nonempty(c.DEFAULT_ROLLING_WINDOWS)
    _assert_iterable_nonempty(c.DEFAULT_ROLLING_STATS)
    _assert_iterable_nonempty(c.DEFAULT_CALENDAR_FEATURES)


def test_allowed_sets_contain_expected_core_values() -> None:
    import eb_features.panel.constants as c

    # Rolling stats should at least include these core operations
    core_stats = {"mean", "std", "min", "max", "sum", "median"}
    assert core_stats.issubset(set(c.ALLOWED_ROLLING_STATS))

    # Calendar features should at least include these
    core_calendar = {"hour", "dow", "dom", "month", "is_weekend"}
    assert core_calendar.issubset(set(c.ALLOWED_CALENDAR_FEATURES))


def test_defaults_are_subsets_of_allowed_sets() -> None:
    import eb_features.panel.constants as c

    # If defaults ever include unsupported values, that should be caught here.
    assert set(c.DEFAULT_ROLLING_STATS).issubset(set(c.ALLOWED_ROLLING_STATS))
    assert set(c.DEFAULT_CALENDAR_FEATURES).issubset(set(c.ALLOWED_CALENDAR_FEATURES))


def test_default_steps_and_windows_are_positive_ints() -> None:
    import eb_features.panel.constants as c

    for k in c.DEFAULT_LAG_STEPS:
        assert isinstance(k, int)
        assert k > 0

    for w in c.DEFAULT_ROLLING_WINDOWS:
        assert isinstance(w, int)
        assert w > 0
