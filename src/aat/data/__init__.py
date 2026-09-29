"""Dataset indexing, leakage-free splitting and multi-center window sampling.

Public entry points:

* :func:`aat.data.build_dataset_index` — scan a data root of labeled protocol
  samples, validate manifests/labels/digests, group by composition/preset/
  sample_origin and assign whole groups to train/val/test with an explicit seed.
* :func:`aat.data.verify_dataset` — re-check an index against its data root,
  collecting every problem.
* :func:`aat.data.sample_batch` — draw a deterministic multi-center batch; one
  block per song with fixed source columns, center activity, source-existence
  and validity/padding masks.

The package only uses the shared ``aat.contracts`` protocol, ``aat.labels``
readers and ``aat.windowing`` helpers; it never loads audio through a training
framework and never mixes two songs in one sample.
"""

from __future__ import annotations

from .batch import DatasetBatch, SourceWindowBlock, sample_batch
from .errors import DatasetError
from .index import (
    DEFAULT_DURATION_TOLERANCE_SECONDS,
    DEFAULT_RATIOS,
    INDEX_VERSION,
    MANIFEST_FILENAME,
    DatasetIndex,
    DatasetVerification,
    SampleEntry,
    build_dataset_index,
    dataset_record_sha256,
    discover_samples,
    portable_data_root,
    scan_sample,
    sha256_file,
    verify_dataset,
)
from .split import (
    GROUP_NAMES,
    SPLITS,
    Component,
    asset_keys,
    assign_splits,
    connected_components,
    cross_split_assets,
    mark_oversized,
    normalize_ratios,
)

__all__ = [
    "DEFAULT_DURATION_TOLERANCE_SECONDS",
    "DEFAULT_RATIOS",
    "GROUP_NAMES",
    "INDEX_VERSION",
    "MANIFEST_FILENAME",
    "SPLITS",
    "Component",
    "DatasetBatch",
    "DatasetError",
    "DatasetIndex",
    "DatasetVerification",
    "SampleEntry",
    "SourceWindowBlock",
    "asset_keys",
    "assign_splits",
    "build_dataset_index",
    "connected_components",
    "cross_split_assets",
    "dataset_record_sha256",
    "discover_samples",
    "mark_oversized",
    "normalize_ratios",
    "portable_data_root",
    "sample_batch",
    "scan_sample",
    "sha256_file",
    "verify_dataset",
]
