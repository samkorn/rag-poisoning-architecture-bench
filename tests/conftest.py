"""Pytest configuration shared across the test suite.

Registers the `integration` and `modal` markers so `--strict-markers`
doesn't warn, and so test runs can be sliced via `pytest -m`:

    pytest -m "not integration and not modal"   # unit only
    pytest -m "integration and not modal"       # integration (no Modal)
    pytest -m "modal"                           # Modal only
    pytest                                      # everything

Integration and Modal classes are responsible for skipping themselves
in `setUpClass` if their prerequisites (data on disk, Modal
credentials, OpenAI key) aren't available — see e.g.
`test_experiment.py`.
"""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register the `integration` and `modal` test markers."""
    config.addinivalue_line(
        "markers",
        "integration: requires the dataset on disk (downloaded or "
        "regenerated locally) and an OpenAI API key.",
    )
    config.addinivalue_line(
        "markers",
        "modal: requires Modal credentials. Calls deployed Modal functions "
        "via .remote() from the test process.",
    )
