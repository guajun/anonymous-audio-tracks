"""Depth comparison with identical explicit full-window statistical readout."""
import torch
from torch import nn


class WindowVHead(nn.Module):
    def __init__(self, depth, hidden=128, k=8, e_dim=128):
        super().__init__()
        if depth not in (2, 4, 8):
            raise ValueError('depth must be 2/4/8')
        self.depth, self.k, self.e_dim = depth, k, e_dim
        self.project = nn.Linear(769, hidden)
        self.temporal = nn.Sequential(*[
            layer for _ in range(depth)
            for layer in (nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU())])
        self.output = nn.Linear(3*hidden, k*e_dim)
        nn.init.normal_(self.output.weight, std=.001)
        nn.init.normal_(self.output.bias, std=.001)

    def forward(self, tokens, rms):
        if tokens.shape[1:] != (197,768) or rms.shape != tokens.shape[:2]:
            raise ValueError('full-window head requires all 197 frame tokens')
        with torch.autocast(device_type=tokens.device.type,dtype=torch.bfloat16,enabled=tokens.is_cuda):
            return self._forward(tokens,rms)

    def _forward(self,tokens,rms):
        h = self.project(torch.cat((tokens.float(), rms[...,None]), dim=-1))
        h = self.temporal(h.transpose(1,2)).transpose(1,2)
        center = .75*h[:,98] + .25*h[:,99]
        # Fixed statistics of learned time features explicitly read every token.
        # Same readout at all depths; no learned query, cropped context or CLS.
        mean = h.mean(1)
        deviation = torch.sqrt((h-mean[:,None]).square().mean(1)+1e-8)
        features = torch.cat((center,mean,deviation),-1)
        v = self.output(features).float().reshape(-1,self.k,self.e_dim)
        if hasattr(self, 'amplitude_output'):
            logits = self.amplitude_output(features).float()
            a = logits.abs() if self.amplitude_transform == 'abs' else torch.nn.functional.softplus(logits)
            return v, a
        return v, torch.linalg.vector_norm(v,dim=-1)

    def architecture(self):
        return dict(depth=self.depth,hidden=128,k=self.k,e_dim=self.e_dim,
            parameters=sum(p.numel() for p in self.parameters()),
            convolution_kernel=3,convolution_dilations=[1]*self.depth,
            intermediate_local_receptive_tokens=1+2*self.depth,
            center_interpolated_local_receptive_tokens=2+2*self.depth,
            final_direct_receptive_tokens=197,input_seconds=2.,
            precision='GPU head bfloat16 autocast; V/amplitude/relations/loss float32; parameters float32',
            token_center_span_seconds=1.96,
            readout='center interpolation + full197-token mean and standard deviation',
            depth_comparison='all depths share full-window readout; depth changes capacity and intermediate local mixing')


class WindowSplitHead(WindowVHead):
    """Independent scalar softplus loudness and unchanged 128D identity output."""
    def __init__(self, depth=4, hidden=128, k=8, e_dim=128, amplitude_transform='softplus'):
        super().__init__(depth, hidden, k, e_dim)
        if amplitude_transform not in ('abs', 'softplus'):
            raise ValueError('amplitude transform must be abs or softplus')
        self.amplitude_transform = amplitude_transform
        self.amplitude_output = nn.Linear(3*hidden, k)
        nn.init.zeros_(self.amplitude_output.weight)
        nn.init.constant_(self.amplitude_output.bias, -4.)

    def architecture(self):
        result = super().architecture()
        result.update(amplitude=self.amplitude_transform+'(Linear(384,8)); independent of Z norm',
                      amplitude_transform=self.amplitude_transform,
                      identity='normalize(Z), original Linear(384,8*128)',
                      additional_parameters=3080)
        return result
