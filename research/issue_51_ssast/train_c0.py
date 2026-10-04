"""P1 only, bounded C0 diagnostic then held-out seeds. No association/cycle."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from features import sha256
from head import TemporalVHead
from shape_loss import c0_loss


def boundaries(x):
    active = np.asarray(x) > .001
    edge = np.diff(np.r_[False, active, False].astype(int))
    return np.flatnonzero(edge == 1), np.flatnonzero(edge == -1) - 1


def metrics(a, target, times, scale):
    costs = np.abs(a - target).mean(0)
    distance = np.abs((a - target) / scale)
    huber = np.where(distance < .1, .5 * distance ** 2 / .1, distance - .05).mean(0)
    ious = np.minimum(a, target).sum(0) / np.maximum(a, target).sum(0).clip(1e-12)
    pit_cost = huber + .1 * (1 - ious)
    slot = int(pit_cost.argmin())  # Same shape combination, entire sample.
    pred = a[:, slot]
    truth = target[:, 0]
    active = truth > .001
    denom = np.maximum(pred, truth).sum()
    iou = float(np.minimum(pred, truth).sum() / denom) if denom > 0 else None
    unused = np.arange(a.shape[1]) != slot
    gate = np.where(a > .001, a, 0)
    count = (gate > 0).sum(1)
    edges = {}
    for name, pp, tt in zip(("onset", "offset"), boundaries(pred), boundaries(truth)):
        # If event counts differ, do not hide misses in a nearest-edge mean.
        errors = np.abs(times[pp] - times[tt]) if len(pp) == len(tt) and len(tt) else None
        edges[name] = dict(pred_count=len(pp), reference_count=len(tt),
            median_abs_seconds=None if errors is None else float(np.median(errors)))
    sorted_cost = np.sort(pit_cost)
    return dict(slot=slot, full_scale_mae=float(costs[slot]),
        normalized_mae=float(costs[slot] / scale),
        active_normalized_mae=float(np.abs(pred[active] - truth[active]).mean() / scale),
        area_iou=iou, silence_false_positive_fraction=float((pred[~active] > .001).mean()),
        unused_false_positive_fraction=float((a[:, unused] > .001).mean()),
        source_count_mae=float(np.abs(count - active.astype(int)).mean()),
        raw_unused_area=float(a[:, unused].sum()),
        gate_unused_area=float(gate[:, unused].sum()),
        gate_normalized_mae=float(np.abs(gate[:, slot] - truth).mean() / scale),
        pit_best_second_gap=float(sorted_cost[1] - sorted_cost[0]),
        pit_tied=bool(sorted_cost[1] - sorted_cost[0] <= 1e-4 * max(1., abs(sorted_cost[0]))),
        boundaries=edges)


def aggregate(rows):
    keys = ("normalized_mae", "active_normalized_mae", "area_iou",
            "silence_false_positive_fraction", "unused_false_positive_fraction",
            "source_count_mae", "gate_normalized_mae")
    result = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    for kind in ("onset", "offset"):
        values = [r["boundaries"][kind]["median_abs_seconds"] for r in rows]
        result[kind + "_median_abs_seconds"] = None if any(v is None for v in values) else float(np.median(values))
    return result


def load_data(cache):
    manifest = json.loads((cache / "manifest.json").read_text())
    rows = []
    for entry in manifest["entries"]:
        path = cache / entry["cache"]
        if sha256(path) != entry["cache_sha256"]:
            raise ValueError("cache digest mismatch")
        z = np.load(path, allow_pickle=False)
        rows.append(dict(entry=entry, tokens=torch.tensor(z["tokens"], device="cuda"),
            bypass=torch.tensor(z["token_rms"], device="cuda"),
            target=torch.tensor(z["targets"], device="cuda"), times=z["center_times"],
            oracle=z["mix_rms_oracle"], valid_fraction=z["context_valid_fraction"]))
    train = [row for row in rows if row["entry"]["split"] == "train"]
    active = torch.cat([r["target"][r["target"] > .001] for r in train])
    if not len(active):
        raise ValueError("no active train RMS")
    scale = float(torch.quantile(active, .95))
    return rows, scale, manifest


@torch.no_grad()
def evaluate(model, rows, scale, out=None):
    metrics_rows = []
    for row in rows:
        pieces = []
        for offset in range(0, len(row["target"]), 256):
            tok = row["tokens"][offset:offset + 256]
            bypass = row["bypass"][offset:offset + 256] / scale
            _, a = model(tok, bypass)
            pieces.append(a.cpu().numpy() * scale)
        a = np.concatenate(pieces)
        target = row["target"].cpu().numpy()
        m = metrics(a, target, row["times"], scale)
        m["sample_id"] = row["entry"]["sample_id"]
        full = row["valid_fraction"] == 1
        m["interior_normalized_mae"] = float(np.abs(a[full, m["slot"]] - target[full, 0]).mean() / scale)
        m["boundary_normalized_mae"] = float(np.abs(a[~full, m["slot"]] - target[~full, 0]).mean() / scale)
        metrics_rows.append(m)
        if out is not None:
            np.savez_compressed(out / (m["sample_id"] + ".npz"), times=row["times"],
                raw_a=a, gated_a=np.where(a > .001, a, 0), target=target)
    return dict(aggregate=aggregate(metrics_rows), samples=metrics_rows)


def fit(rows, scale, *, seed, max_steps, seconds, out, diagnostic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = TemporalVHead().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    train = [r for r in rows if r["entry"]["split"] == "train"]
    val = [r for r in rows if r["entry"]["split"] == "val"]
    if diagnostic:
        train, val = train[:1], train[:1]
    out.mkdir(parents=True, exist_ok=True)
    best, best_state, stale = float("inf"), None, 0
    checkpoint_best, updates = float("inf"), 0
    history = []
    start = time.monotonic()
    stop = "max_steps"
    for step in range(1, max_steps + 1):
        if time.monotonic() - start > seconds:
            stop = "wall_clock_budget"
            break
        selected = random.sample(train, min(4, len(train)))
        optimizer.zero_grad(set_to_none=True)
        # Accumulate whole samples separately to bound activation memory.
        loss_value = 0.
        parts = []
        for row in selected:
            _, a = model(row["tokens"], row["bypass"] / scale)
            loss, detail = c0_loss(a, row["target"] / scale)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss; refusing to continue training")
            (loss / len(selected)).backward()
            loss_value += float(loss.detach()) / len(selected)
            parts.append({k: float(v.detach()) if torch.is_tensor(v) else v for k, v in detail.items()})
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        optimizer.step()
        updates = step
        if step % 100 == 0 or step == max_steps:
            vm = evaluate(model, val, scale)
            score = vm["aggregate"]["normalized_mae"] + .1 * (1 - vm["aggregate"]["area_iou"])
            improved = score < best * .99
            if score < checkpoint_best:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                checkpoint_best = score
            if improved:
                best, stale = score, 0
            else:
                stale += 1
            h = dict(step=step, train_loss=loss_value, components=parts,
                val=vm["aggregate"], gradient_norm_before_clip=grad,
                elapsed_seconds=time.monotonic() - start)
            history.append(h)
            print(json.dumps(dict(seed=seed, diagnostic=diagnostic, **h)), flush=True)
            (out / "progress.json").write_text(json.dumps(history, indent=2) + "\n")
            if step >= 1000 and stale >= 10:
                stop = "val_plateau_10_checks"
                break
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    report = dict(seed=seed, diagnostic=diagnostic, steps=updates,
        stop_reason=stop, elapsed_seconds=time.monotonic() - start,
        scale_r=scale, scale_policy="train-only active RMS P95", loss="Huber(delta=.1)/.1 + .1 areaIoU; .1 silence + .1 unused",
        train_gate="raw norm differentiable; threshold1e-3 only in evaluation",
        architecture="frozen frame197x768 -> Linear128 -> 2 temporal conv3 -> center interpolation98.25 -> K8 V128",
        history=history, splits={})
    splits = ("train",) if diagnostic else ("train", "val", "test")
    for split in splits:
        subset = train if diagnostic else [r for r in rows if r["entry"]["split"] == split]
        report["splits"][split] = evaluate(model, subset, scale, out)
    torch.save(best_state, out / "head.pth")
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--diagnostic-steps", type=int, default=1000)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--seconds-per-seed", type=int, default=3600)
    args = p.parse_args()
    torch.set_num_threads(4)
    rows, scale, manifest = load_data(args.cache)
    args.out.mkdir(parents=True, exist_ok=True)
    # Baselines are written before fitting, with exactly the cached target grid.
    baseline = {}
    for kind in ("oracle", "zero"):
        baseline[kind] = {}
        for split in ("train", "val", "test"):
            rr = []
            for row in rows:
                if row["entry"]["split"] != split:
                    continue
                a = np.zeros((len(row["target"]), 8))
                if kind == "oracle":
                    a[:, 0] = row["oracle"]
                rr.append(metrics(a, row["target"].cpu().numpy(), row["times"], scale))
            baseline[kind][split] = aggregate(rr)
    (args.out / "baselines.json").write_text(json.dumps(baseline, indent=2) + "\n")
    diag = fit(rows, scale, seed=46, max_steps=args.diagnostic_steps, seconds=3600,
               out=args.out / "overfit-seed46", diagnostic=True)
    dm = diag["splits"]["train"]["aggregate"]
    diag_pass = dm["normalized_mae"] <= .05 and dm["area_iou"] >= .90
    summary = dict(stage="P1", diagnostic_passed=diag_pass,
        input_cache_manifest_sha256=sha256(args.cache / "manifest.json"),
        backbone=manifest["encoder"], diagnostic_metrics=dm, results=[])
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not diag_pass:
        print("P1 STOP: single-sample fitting gate failed; no held-out training/P2", flush=True)
        return
    for seed in (46, 47, 48):
        result = fit(rows, scale, seed=seed, max_steps=args.steps,
                     seconds=args.seconds_per_seed, out=args.out / f"seed{seed}")
        summary["results"].append(dict(seed=seed, steps=result["steps"],
            stop_reason=result["stop_reason"], elapsed_seconds=result["elapsed_seconds"],
            splits={k: v["aggregate"] for k, v in result["splits"].items()}))
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary["p2_gate_passed"] = all(
        r["splits"]["val"]["normalized_mae"] <= .1 and
        r["splits"]["val"]["area_iou"] >= .8 and
        r["splits"]["val"]["silence_false_positive_fraction"] <= .01 and
        r["splits"]["val"]["unused_false_positive_fraction"] <= .01 and
        all(r["splits"]["val"][k + "_median_abs_seconds"] is not None and
            r["splits"]["val"][k + "_median_abs_seconds"] <= .04 for k in ("onset", "offset"))
        for r in summary["results"])
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
