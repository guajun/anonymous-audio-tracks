import numpy as np
import pytest
from aat.envelopes.labels import EnvelopeData
from aat.envelopes.local_context import audit_local_context
from aat.envelopes.curriculum import c1_local_config


def labels_for_events(events,*,origin=0.):
    times=origin+np.arange(651)*.02
    y=np.zeros((len(times),len(events)),dtype=np.float32)
    for source,event_times in enumerate(events):
        for start,end in event_times:
            y[(times>=origin+start-1e-9)&(times<=origin+end+1e-9),source]=.1
    valid=np.ones(len(times),dtype=bool);valid[-1]=False
    return EnvelopeData(times,y,valid,tuple(f's{i}' for i in range(len(events))),1000,.02,.02,{})


def dense_labels(origin=0.):
    return labels_for_events([[(.4+.64*i,.54+.64*i) for i in range(18)],
                              [(.72+.64*i,.86+.64*i) for i in range(18)]],origin=origin)


def test_actual_two_second_windows_cover_notes_and_silence_bridges():
    labels=dense_labels()
    original=labels.rms.copy()
    centers=np.arange(25,276)*.04
    report,arrays=audit_local_context(labels,centers,audio_frames=13000)
    assert report['passed'] and report['full_context_centers']==251
    assert arrays['complete_event_counts'].min()>=2
    assert np.all(arrays['active_neighbor_covered'][arrays['center_active']])
    assert np.all(arrays['bridge_endpoints_covered'][arrays['silence_bridge']])
    assert all(s['bridge_targets_exactly_zero']>0 for s in report['sources'])
    np.testing.assert_array_equal(labels.rms,original)


def test_long_segment_cannot_replace_local_input_evidence():
    labels=labels_for_events([[(1.2,1.4),(6.8,7.0)]])
    centers=np.arange(30,176)*.04
    report,_=audit_local_context(labels,centers,audio_frames=13000)
    assert not report['passed']
    assert report['sources'][0]['active_centers_with_adjacent_evidence']==0
    assert report['sources'][0]['silence_bridge_centers_with_both_endpoints']==0


def test_boundary_guard_rejects_partial_contextual_note():
    labels=labels_for_events([[(4.4,4.54),(5.9,6.04)]])
    report,_=audit_local_context(labels,np.array([5.0]),audio_frames=13000)
    assert not report['passed']
    assert report['sources'][0]['minimum_complete_events_per_window']==1


def test_padded_input_is_not_counted_as_real_context():
    labels=dense_labels()
    report,_=audit_local_context(labels,np.array([.4]),audio_frames=13000)
    assert not report['passed'] and report['full_context_centers']==0
    assert any(v['reason']=='input_requires_padding' for v in report['violations_first_20'])


def test_absolute_track_origin_is_preserved():
    labels=dense_labels(origin=100.)
    centers=100.+np.arange(25,276)*.04
    report,arrays=audit_local_context(labels,centers,audio_frames=13000,origin_seconds=100.)
    assert report['passed']
    np.testing.assert_allclose(arrays['input_bounds_seconds'][0],[100.,102.])
    assert labels.center_times[0]==100.


def test_new_config_retains_alternation_and_distinct_events():
    starts=set()
    for seed in range(4640,4650):
        config,onsets,owners=c1_local_config(seed,str(seed))
        assert len(onsets)==36 and len(config.sources)==2
        assert all(a!=b for a,b in zip(owners,owners[1:]))
        assert np.diff(onsets).min()>=.28-1e-9
        assert all(len(s.pattern.notes)==18 for s in config.sources)
        starts.add(owners[0])
    assert starts=={0,1}


def test_centers_must_belong_to_valid_label_grid():
    with pytest.raises(ValueError,match='label grid'):
        audit_local_context(dense_labels(),np.array([1.001]),audio_frames=13000)
