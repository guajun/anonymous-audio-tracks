"""Session fixtures for the training tests.

The smoke corpus is written once per session: real protocol documents and real
PCM audio, three disjoint asset families, no DawDreamer and no weights.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def smoke_corpus(tmp_path_factory):
    from aat.training.dataset import make_smoke_dataset

    root = tmp_path_factory.mktemp("smoke-corpus") / "dataset"
    return make_smoke_dataset(root, seed=20260929)
