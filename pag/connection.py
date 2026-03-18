import torch
from torch import nn
from torch.nn import LayerNorm
import torch.nn.functional as F


class HyperConnection(nn.Module):
    def __init__(
        self,
        channels: int,
        expansion_rate: int,
        layer_id: int = 0,
        dynamic: bool = True,
        device=None,
    ):
        super(HyperConnection, self).__init__()

        self.rate = expansion_rate
        self.layer_id = layer_id
        self.dynamic = dynamic

        self.static_beta = nn.Parameter(torch.ones((expansion_rate,), device=device))

        init_alpha0 = torch.zeros((expansion_rate, 1), device=device)
        init_alpha0[layer_id % expansion_rate, 0] = 1.0
        self.static_alpha = nn.Parameter(
            torch.cat([init_alpha0, torch.eye((expansion_rate), device=device)], dim=1)
        )

        if self.dynamic:
            self.dynamic_alpha_fn = nn.Parameter(
                torch.zeros((channels, expansion_rate + 1), device=device)
            )
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(1, device=device) * 0.01)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros((channels,), device=device))
            self.dynamic_beta_scale = nn.Parameter(torch.ones(1, device=device) * 0.01)
            self.layer_norm = LayerNorm(channels)

    def width_connection(self, h):
        # get alpha and beta
        if self.dynamic:
            norm_h = self.layer_norm(h)

        if self.dynamic:
            wc_weight = norm_h @ self.dynamic_alpha_fn
            wc_weight = F.tanh(wc_weight)
            dynamic_alpha = wc_weight * self.dynamic_alpha_scale
            alpha = dynamic_alpha + self.static_alpha[None, None, ...]
        else:
            alpha = self.static_alpha[None, None, ...]

        if self.dynamic:
            dc_weight = norm_h @ self.dynamic_beta_fn
            dc_weight = F.tanh(dc_weight)
            dynamic_beta = dc_weight * self.dynamic_beta_scale
            beta = dynamic_beta + self.static_beta[None, None, ...]
        else:
            beta = self.static_beta[None, None, ...]

        # width connection
        mix_h = alpha.transpose(-1, -2) @ h

        return mix_h, beta

    def depth_connection(self, mix_h, h_o, beta):
        h = torch.einsum("blh,bln->blnh", h_o, beta) + mix_h[..., 1:, :]

        return h
