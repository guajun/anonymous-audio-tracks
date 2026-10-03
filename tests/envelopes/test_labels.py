"""Continuous values, source time alignment, and independent artifact semantics."""
import numpy as np
import pytest
from aat.envelopes.labels import label_sample, EnvelopeData, overlap_ratio
from tests.labels.support import make_sample_dir


def test_continuous_envelope_roundtrip_and_time_origin(tmp_path):
    rate=16000
    t=np.arange(rate*2)/rate
    envelope=np.minimum(t,1.0)*np.maximum(2.0-t,0.0)
    tone=.3*envelope*np.sin(2*np.pi*400*t)
    root=make_sample_dir(tmp_path/"sample",stems={"s01":tone},track_start_seconds=12.0)
    result=label_sample(root)
    restored=EnvelopeData.load(root)
    np.testing.assert_array_equal(restored.rms,result.rms)
    assert restored.center_times[0]==12.0
    assert len(np.unique(np.round(result.rms[:,0],4)))>30
    assert not restored.valid[-1]
    assert (root/"activity.json").exists() is False


def test_no_per_source_normalization_and_channel_phase_cancellation(tmp_path):
    tone=np.sin(np.arange(32000)*2*np.pi*400/16000)*.2
    stereo=np.stack((tone,-tone),axis=1)
    root=make_sample_dir(tmp_path/"sample",stems={"s01":stereo,"s02":stereo*.5})
    labels=label_sample(root)
    assert labels.rms[20,0]>.1
    assert labels.rms[20,1]/labels.rms[20,0]==pytest.approx(.5,rel=.002)
    assert overlap_ratio(labels.rms)==1.0


def test_corrupt_stem_rejected(tmp_path):
    root=make_sample_dir(tmp_path/"sample",stems={"s01":np.ones(32000)*.1})
    with (root/"stems/s01.wav").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError,match="digest"):
        label_sample(root)


def test_corrupt_arrays_rejected(tmp_path):
    root=make_sample_dir(tmp_path/"sample",stems={"s01":np.ones(32000)*.1})
    label_sample(root)
    with (root/"envelope.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError,match="digest"):
        EnvelopeData.load(root)


def test_c0_render_config_has_one_source_and_safe_note_spacing():
    from aat.envelopes.curriculum import c0_config
    from aat.render.timeline import expand_score
    config=c0_config(4600,"c0-test")
    events=expand_score(config)
    assert len(config.sources)==1
    assert len(events)==3
    assert min(b.start_seconds-a.start_seconds for a,b in zip(events,events[1:]))>1.5+config.sources[0].amp.release_ms/1000
