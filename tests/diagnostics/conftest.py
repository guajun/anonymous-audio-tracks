"""Session fixtures for the issue #24 diagnostics tests.

The smoke corpus is the same synthetic NumPy PCM corpus used by the training
tests (no DawDreamer, no weights).  The sampling-coverage diagnostic consumes
it as engineering evidence, never as model evidence.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def smoke_corpus(tmp_path_factory):
    from aat.training.dataset import make_smoke_dataset

    root = tmp_path_factory.mktemp("issue24-diagnostics") / "corpus"
    return make_smoke_dataset(root, seed=20260929)
