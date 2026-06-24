"""Hyperbolic Intensity Head (HIH).

We embed the fused representation ``z`` into the Lorentz model of hyperbolic
space and interpret sentiment as a *radial* quantity:

    * intensity  = geodesic radius from the origin   (||z|| under expmap_0)
    * polarity   = a learned direction (sign)        (tanh(w . z))
    * prediction = 3 * polarity * r / (r + 1)         in [-3, 3]

The exponential volume growth of hyperbolic space gives the sparse, high-arousal
samples (e.g. Highly-Negative / Highly-Positive in MOSI) more separable room,
mitigating the long-tail collapse observed in Euclidean regression heads.

Numerical stability is handled with clamping on ``arccosh`` arguments and norm
clipping before the exponential map, as recommended for Lorentz operations.
"""
import torch
import torch.nn as nn


# ---- numerically-stable Lorentz primitives (curvature c > 0) ----

def _safe_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8,
               max_norm: float = 5.0) -> torch.Tensor:
    norm = x.norm(dim=dim, keepdim=True).clamp_min(eps)
    return norm.clamp_max(max_norm)


def lorentz_expmap0(v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Exponential map at the origin of the Lorentz hyperboloid.

    Args:
        v: tangent (spatial) vector of shape (..., d).
        c: positive curvature scalar tensor.
    Returns:
        Point on the hyperboloid of shape (..., d + 1) with the time coordinate
        in index 0.
    """
    sqrt_c = c.clamp_min(1e-6).sqrt()
    v_norm = _safe_norm(v)                       # (..., 1)
    scaled = (sqrt_c * v_norm).clamp(max=20.0)   # avoid sinh/cosh overflow
    x_time = torch.cosh(scaled) / sqrt_c         # (..., 1)
    x_space = torch.sinh(scaled) * v / (v_norm * sqrt_c)
    return torch.cat([x_time, x_space], dim=-1)


def lorentz_radius(v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Geodesic distance from ``expmap0(v)`` to the origin.

    For the Lorentz model this equals ``||v||`` exactly, but we compute it via
    the manifold definition (and clamp) so the value stays consistent with the
    embedding actually used and is safe to differentiate.
    """
    sqrt_c = c.clamp_min(1e-6).sqrt()
    v_norm = _safe_norm(v).squeeze(-1)           # (...,)
    # arccosh(cosh(sqrt_c * ||v||)) / sqrt_c == ||v||, kept explicit for clarity
    arg = torch.cosh((sqrt_c * v_norm).clamp(max=20.0)).clamp_min(1.0 + 1e-6)
    return torch.acosh(arg) / sqrt_c


class HyperbolicIntensityHead(nn.Module):
    def __init__(self, in_dim: int, curvature: float = 1.0,
                 learn_curvature: bool = True):
        super().__init__()
        self.proj = nn.Linear(in_dim, in_dim)
        self.polarity = nn.Linear(in_dim, 1)
        c_init = torch.tensor(float(curvature))
        if learn_curvature:
            # parametrise via softplus to keep c > 0
            self._c_raw = nn.Parameter(torch.log(torch.expm1(c_init)))
        else:
            self.register_buffer("_c_fixed", c_init)
        self.learn_curvature = learn_curvature

    @property
    def curvature(self) -> torch.Tensor:
        if self.learn_curvature:
            return torch.nn.functional.softplus(self._c_raw) + 1e-4
        return self._c_fixed

    def forward(self, z: torch.Tensor):
        """Returns (y_hat, radius, point_on_manifold)."""
        c = self.curvature
        v = self.proj(z)
        radius = lorentz_radius(v, c)                      # intensity  (B,)
        polarity = torch.tanh(self.polarity(z)).squeeze(-1)  # in (-1, 1)  (B,)
        y_hat = 3.0 * polarity * radius / (radius + 1.0)
        point = lorentz_expmap0(v, c)                      # for visualisation
        return y_hat, radius, point
