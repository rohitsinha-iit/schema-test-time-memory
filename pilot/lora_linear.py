"""LoRA wrapper for a frozen nn.Linear (no transformers dependency)."""
import torch


class LoRALinear(torch.nn.Module):
    def __init__(self, base, r=16, alpha=32):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = torch.nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / base.in_features ** 0.5)
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        self.enabled = True

    def forward(self, x):
        if not self.enabled:
            return self.base(x)
        return self.base(x) + (x.float() @ self.A.T @ self.B.T * self.scale).to(x.dtype)
