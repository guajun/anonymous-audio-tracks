"""Reusable curriculum configuration and real post-effect stem supervision."""
import numpy as np
import pytest

from aat.data.curriculum import c1_config, prepare_c0, prepare_c1, prepare_c1_local
from aat.labels.envelope import EnvelopeData


def test_c1_randomized_start_and_recurrence():
    first_owners = set()
    for seed in range(4610, 4620):
        config, onsets, owners = c1_config(seed, str(seed))
        assert len(config.sources) == 2 and len(onsets) == 5
        assert all(a != b for a, b in zip(owners, owners[1:]))
        assert owners[0] == owners[2] == owners[4]
        assert np.diff(onsets).min() >= 2.6 - 1e-9
        first_owners.add(owners[0])
    assert first_owners == {0, 1}


@pytest.mark.parametrize('prepare', [prepare_c0, prepare_c1, prepare_c1_local])
@pytest.mark.parametrize('counts', [(1, 2), (1, -1, 0), (0, 0, 0), (True, 0, 0)])
def test_invalid_counts_do_not_create_output(tmp_path, prepare, counts):
    target = tmp_path / 'invalid'
    with pytest.raises(ValueError):
        prepare(target, counts=counts)
    assert not target.exists()


@pytest.mark.parametrize('prepare', [prepare_c0, prepare_c1, prepare_c1_local])
def test_existing_data_is_not_overwritten(tmp_path, prepare):
    marker = tmp_path / 'existing'
    marker.write_text('retain')
    with pytest.raises(ValueError, match='empty'):
        prepare(tmp_path, counts=(1, 0, 0))
    assert marker.read_text() == 'retain'


@pytest.mark.integration
@pytest.mark.parametrize('prepare', [prepare_c0, prepare_c1, prepare_c1_local])
def test_real_rendered_curriculum_and_envelopes(tmp_path, prepare):
    pytest.importorskip('dawdreamer')
    index = prepare(tmp_path / 'corpus', counts=(1, 0, 0))
    assert len(index['entries']) == 1
    entry = index['entries'][0]
    sample = tmp_path / 'corpus' / entry['directory']
    labels = EnvelopeData.load(sample)
    assert labels.rms.max() > .002
    assert np.any(labels.rms == 0)
    assert entry['acoustic_overlap_ratio'] == 0
    assert entry['stem_sum']['max_abs_error_lsb'] <= entry['stem_sum']['tolerance_lsb']
    if index['stage'] == 'C1-local':
        assert entry['local_context']['passed']
        assert all(s['event_count'] == 18 for s in entry['local_context']['sources'])
