"""Query-free temporal V head. No hard gate in the training gradient path."""
import torch
from torch import nn


class TemporalVHead(nn.Module):
    def __init__(self, dim=768, hidden=128, k=8, e_dim=128):
        super().__init__()
        self.k, self.e_dim = k, e_dim
        self.project = nn.Linear(dim + 1, hidden)
        self.temporal = nn.Sequential(nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.GELU(), nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU())
        self.output = nn.Linear(hidden, k * e_dim)
        nn.init.normal_(self.output.weight, std=.001)
        nn.init.normal_(self.output.bias, std=.001)

    def forward(self, tokens, rms):
        # [windows,197,768], bypass in common train-only RMS scale.
        if tokens.ndim != 3 or tokens.shape[1] not in (197, 6) or tokens.shape[2] != 768 or rms.shape != tokens.shape[:2]:
            raise ValueError("head requires audited197-token frame grid or its exact96..101 subset")
        # Two kernel3 convolutions have radius2. The two interpolated center
        # states need exactly input tokens96..101; these encoder tokens already
        # attend to the full 2s window. This equals the full-sequence head at
        # the center and avoids computing discarded output positions.
        if tokens.shape[1] == 197:
            tokens = tokens[:, 96:102]
            rms = rms[:, 96:102]
        tokens = tokens.float()
        h = self.project(torch.cat([tokens, rms.unsqueeze(-1)], dim=-1))
        h = self.temporal(h.transpose(1, 2)).transpose(1, 2)
        # frame stride1 centers: 0.0175+.01*j -> 1s at j=98.25.
        center = .75 * h[:, 2] + .25 * h[:, 3]
        v = self.output(center).reshape(-1, self.k, self.e_dim)
        return v, torch.linalg.vector_norm(v, dim=-1)
