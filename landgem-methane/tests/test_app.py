from pathlib import Path

import pytest

APP = str(Path(__file__).resolve().parents[1] / "app.py")

st_testing = pytest.importorskip("streamlit.testing.v1")


def _run(setup=None):
    at = st_testing.AppTest.from_file(APP, default_timeout=120).run()
    if setup:
        setup(at)
        at.run()
    return at


def test_app_starts_without_errors():
    at = _run()
    assert not at.exception
    assert len(at.tabs) == 6


def test_app_demo_observations_and_conversion():
    def setup(at):
        for c in at.checkbox:
            if c.label in ("Use synthetic demo observations", "Convert generation to surface emission"):
                c.check()
    at = _run(setup)
    assert not at.exception
    assert any(m.label == "Observations used" for m in at.metric)


def test_app_preset_sets_parameters():
    def setup(at):
        [s for s in at.selectbox if s.label == "Start from a preset"][0].set_value("EPA AP-42 inventory (k 0.04, L0 100)")
    at = _run(setup)
    assert not at.exception
    assert at.session_state.k == 0.04 and at.session_state.L0 == 100.0


def test_app_rejects_inconsistent_years():
    def setup(at):
        [n for n in at.number_input if n.label == "Last year of waste"][0].set_value(1990)
    at = _run(setup)
    assert not at.exception and len(at.error) == 1
