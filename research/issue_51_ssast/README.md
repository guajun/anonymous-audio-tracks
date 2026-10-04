# Issue 51 execution

Research code only; shared curriculum/envelope/local-context modules stay unchanged.
P0 and bounded P1 C0 entry points are available; no TimeCycleLoss or P2 association.

P0 uses the fixed official SSAST source commit
`a1a3eecb94731e226308a6812f2fbf268d789caf` downloaded to a run directory.
Load every official pretraining key strictly before extracting downstream
unpooled transformer tokens. Frequency aggregation/time pooling is not done
by this probe.

Example (dedicated remote checkout):

```sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv-p0/bin/python research/issue_51_ssast/probe.py \
  --source runs/p0/ast_models.py --checkpoint /path/to/official/frame.pth \
  --model-type frame --out runs/p0/frame-probe
```

A patch run is a separate diagnostic and does not pass the frame stage gate.
An unverified converted mirror is not an official checkpoint and cannot pass
the provenance gate merely by having matching tensor dimensions.
The user subsequently supplied a frame checkpoint, whose local/remote SHA256
matches and whose full state strictly loads. Declare `--checkpoint-origin
user-provided` to record this origin explicitly; this is not an independent
authentication of the author's original download.

P1 runs:

```sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv-p0/bin/python research/issue_51_ssast/dataset_adapter.py \
  --source runs/p0/ast_models.py --checkpoint runs/p0/user-frame.pth --out runs/p1/frame-c0-cache
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv-p0/bin/python research/issue_51_ssast/train_c0.py \
  --cache runs/p1/frame-c0-cache --out runs/p1/frame-c0-fit
```

The cache retains all197 temporal tokens at every20ms prediction center.
Two kernel3 convolutions followed by center interpolation need input tokens
96..101; computing only this receptive field is numerically tested against
the full-sequence head. Each frozen token still attends to the full2s input.
Training has no hard gate; evaluation reports raw and thresholded amplitudes.
Single-sample fitting is capped at1000 updates, then seeds46/47/48 at5000 or
one hour each, with validation early stopping. Test predictions are generated
once after checkpoint selection. Acceptance remains fixed; diagnostic threshold
sweeps do not change the stage gate or tune model settings on test data.
JSON reports and token arrays are written to the run directory; checkpoint,
environment, rendering, and caches are never committed.
