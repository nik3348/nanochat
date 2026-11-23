"""
DeepCrossAttention (Heddes et al. 2025, arXiv:2502.06785) building blocks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def dca_slots(n_pushed: int, k: int) -> int:
    """Static number of stack entries seen by a GRN-v3."""
    return 1 + int(n_pushed > k) + min(n_pushed, k)


class DCAStack:
    """x0 + running sum of middle outputs + last k sublayer outputs."""

    def __init__(self, x0: torch.Tensor, k: int):
        self.x0, self.k = x0, k
        self.mid = None
        self.recent = []

    def push(self, y: torch.Tensor):
        self.recent.append(y)
        if len(self.recent) > self.k:
            ev = self.recent.pop(0)
            self.mid = ev if self.mid is None else self.mid + ev

    def tensors(self):
        return [self.x0] + ([self.mid] if self.mid is not None else []) + self.recent


class GRN(nn.Module):
    """n_out independent GRN-v3 instances over the same depth stack."""

    def __init__(self, n_out: int, n_slots: int, n_embd: int):
        super().__init__()
        self.n_out, self.n_slots, self.n_embd = n_out, n_slots, n_embd
        self.b = nn.Parameter(torch.ones(n_out, n_slots, n_embd))
        self.w = nn.Parameter(torch.zeros(n_out, n_embd))

    @torch.no_grad()
    def reset_parameters(self, w_std: float = 1e-3):
        self.b.fill_(1.0)
        nn.init.normal_(self.w, mean=0.0, std=w_std)

    def forward(self, stack):
        assert len(stack) == self.n_slots, f"expected {self.n_slots} stack entries, got {len(stack)}"
        out = 0
        for s, g in enumerate(stack):
            a = F.relu(F.linear(g, self.w.to(g.dtype)))
            coef = self.b[:, s, :].to(g.dtype)[None, None] + a.unsqueeze(-1)
            out = out + g.unsqueeze(2) * coef
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, C, L = 2, 5, 16, 6
    for k in (0, 1, 2, 4, 100):
        x0 = torch.randn(B, T, C)
        stack = DCAStack(x0, k)
        resid = x0.clone()
        lin = [nn.Linear(C, C) for _ in range(2 * L)]
        grns = [GRN(1, dca_slots(i, k), C) for i in range(2 * L + 1)]
        for grn in grns:
            grn.reset_parameters(w_std=0.0)
        for i in range(2 * L):
            gin = grns[i](stack.tensors())[:, :, 0]
            assert torch.allclose(gin, resid, atol=1e-5), (k, i)
            y = lin[i](F.rms_norm(gin, (C,)))
            stack.push(y)
            resid = resid + y
        gout = grns[-1](stack.tensors())[:, :, 0]
        assert torch.allclose(gout, resid, atol=1e-5)

        g = GRN(3, dca_slots(4, 2), C)
        g.reset_parameters()
        st = DCAStack(torch.randn(B, T, C), 2)
        for _ in range(4):
            st.push(torch.randn(B, T, C))
        g(st.tensors()).sum().backward()
        assert g.w.grad.abs().sum() > 0
    print("OK: GRN-v3 == residual sum at init for all k; w receives gradient")
