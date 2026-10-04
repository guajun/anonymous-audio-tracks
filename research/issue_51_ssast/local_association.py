"""Finite recent-event association. Cycle is not part of execution or loss."""
import torch


def sinkhorn(scores, iterations=8):
    z = scores
    for _ in range(iterations):
        z = z - torch.logsumexp(z, dim=-1, keepdim=True)
        z = z - torch.logsumexp(z, dim=-2, keepdim=True)
    return z.exp()


def transport(v, times, *, gate=.001, horizon=.9, temperature=.1):
    # v[T,K,D] in full-scale units. E strictly exists only where A>gate.
    # Gate selects reliable nonzero evidence, not a persistent silence identity.
    a = torch.linalg.vector_norm(v, dim=-1)
    valid = (a.detach() > gate)
    e = torch.where(valid[..., None], v / a.clamp_min(1e-12)[..., None], torch.zeros_like(v))
    t, k, d = v.shape
    tracks = [None] * k  # latest genuine candidate descriptor within horizon
    latest = [-1] * k
    fragment_ids = [-1] * k
    next_fragment = 0
    fragment_history = []
    amplitudes, matrices, fragments, anchors = [], [], [], []
    for i in range(t):
        ref_rows = [j for j in range(k) if tracks[j] is not None and
                    float(times[i] - times[latest[j]]) <= horizon]
        for j in range(k):
            if tracks[j] is not None and j not in ref_rows:
                tracks[j], latest[j] = None, -1
                fragment_ids[j] = -1
                fragments.append([i, j])
        current = torch.nonzero(valid[i], as_tuple=False).flatten().tolist()
        c = torch.zeros(k, k, device=v.device, dtype=v.dtype)
        if not current:
            amplitudes.append(a[i] * 0)
            matrices.append(c)
            fragment_history.append(fragment_ids.copy())
            continue
        if not ref_rows:
            # First genuine frame: direct anchor; no cosine self-match.
            c[current, current] = 1.
            for j in current:
                tracks[j], latest[j] = e[i, j], i
                fragment_ids[j] = next_fragment
                next_fragment += 1
                anchors.append([i, j])
        else:
            refs = torch.stack([tracks[j] for j in ref_rows])
            refs = refs / refs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            similarities = refs @ e[i, current].T
            # Pad to KxK with empty matching positions. Dummy rows carry no E;
            # they absorb genuinely new candidates into available anonymous slots.
            scores = torch.zeros(k, k, device=v.device, dtype=v.dtype)
            scores[:len(ref_rows), :len(current)] = similarities / temperature
            weights = sinkhorn(scores)
            free = [j for j in range(k) if j not in ref_rows]
            row_order = ref_rows + free
            # Current columns are explicit, inactive columns have no amplitude.
            for ri, j in enumerate(row_order):
                c[j, current] = weights[ri, :len(current)]
            transported = c @ a[i]
            for j in range(k):
                if float(transported[j].detach()) > gate:
                    if tracks[j] is None:
                        fragment_ids[j] = next_fragment
                        next_fragment += 1
                    # Replace with descriptor from this actual frame; no EMA,
                    # no initial/frozen/unbounded acoustic prototype.
                    # Store the newest actual nonzero candidate, not an
                    # recursively mixed descriptor. The discrete reference
                    # selection is separate from differentiable C->A transport;
                    # current and referenced genuine E still receive shape grads.
                    picked = int(c[j].detach().argmax())
                    tracks[j] = e[i, picked]
                    latest[j] = i
        amplitudes.append(c @ a[i])
        matrices.append(c)
        fragment_history.append(fragment_ids.copy())
    return torch.stack(amplitudes), torch.stack(matrices), dict(
        anchors=anchors, expired_fragments=fragments, fragment_ids_by_frame=fragment_history,
        fragment_count=next_fragment, horizon_seconds=horizon,
        temperature=temperature, gate=gate, sinkhorn_iterations=8,
        memory='latest actual nonzero candidate, dominant-match reference selection; expires after horizon; no EMA/recursive mixture',
        zero_e='undefined and excluded', cycle_enabled=False)
