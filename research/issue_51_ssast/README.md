# Issue 51 execution

Research code only; shared curriculum/envelope/local-context modules stay unchanged.
No training entry point and no TimeCycleLoss.

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
JSON reports and token arrays are written to the run directory; checkpoint,
environment, rendering, and caches are never committed.
