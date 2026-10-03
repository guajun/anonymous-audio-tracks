import numpy as np
import pytest
from aat.envelopes.metrics import event_diagnostics


def test_boundary_offsets_stay_on_original_time_axis():
    y=np.array([0,0,.1,.1,.1,0,0,0])
    p=np.roll(y,1)
    metrics=event_diagnostics(p,y,np.arange(8)*.04)
    assert metrics["matched_events"]==1
    assert metrics["onset_signed_seconds"]==pytest.approx(.04)
    assert metrics["offset_mae_seconds"]==pytest.approx(.04)


def test_merged_prediction_cannot_match_two_reference_events():
    y=np.array([0,.1,.1,0,0,.1,.1,0])
    p=np.array([0,.1,.1,.1,.1,.1,.1,0])
    metrics=event_diagnostics(p,y,np.arange(8)*.04)
    assert metrics["matched_events"]==1
    assert metrics["missed_events"]==1


def test_fully_missed_events_do_not_fabricate_zero_boundary_error():
    y=np.array([0,.1,.1,0,.1,0])
    metrics=event_diagnostics(np.zeros(6),y,np.arange(6)*.04)
    assert metrics["missed_events"]==2
    assert metrics["onset_mae_seconds"] is None
    assert metrics["offset_mae_seconds"] is None
    assert metrics["frame_f1"]==0
