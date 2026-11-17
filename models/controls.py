# controls.py
# Matched controls for the rebuttal: the same DLinear backbone, enhancers and two-side ensemble as MAGNet+DLinear,
# with the graph removed (DLinear-only) or replaced by a plain exogenous MLP of about the same capacity.

import torch
import torch.nn as nn


class ExoMLPDLinear(nn.Module):
    """
    DLinear (+ the cell's enhancers) plus a simple exogenous fusion: the flattened exo history goes through a 2-layer
    MLP straight to the horizon and is added to the backbone forecast. The hidden width is chosen so the MLP has about
    as many parameters as the MAGNet graph branch it replaces. The last layer starts at zero, so training starts from
    plain DLinear (like MAGNet's cold-start gates).

    base : a MAGNETWithDLinear in mode 3 (backbone only)
    """

    def __init__(self, base, n_exo, seq_len, horizon, target_params):
        super().__init__()
        self.base = base
        self.magnet_graph = None
        hidden = max(8, round(target_params / (seq_len * n_exo + 1 + horizon)))
        self.exo_mlp = nn.Sequential(nn.Linear(seq_len * n_exo, hidden), nn.GELU(), nn.Linear(hidden, horizon))
        nn.init.zeros_(self.exo_mlp[-1].weight)
        nn.init.zeros_(self.exo_mlp[-1].bias)

    def forward(self, exo_patches, endo_patches):
        out = self.base(exo_patches, endo_patches)
        squeeze = out.dim() == 3
        out = out.squeeze(-1) if squeeze else out
        out = out + self.exo_mlp(exo_patches.permute(0, 3, 1, 2).flatten(1))   # [B, n_exo * seq_len]
        return out.unsqueeze(-1) if squeeze else out


class XAttnDLinear(nn.Module):
    """
    Standalone "plain cross-attention" control (rebuttal, 2026-11-12): DLinear (+ the cell's enhancers, same two-side
    ensemble) plus ONE standard cross-attention block from the target's patches to the exogenous patches -- no graph
    (no exo-exo GAT, no node construction), no Granger weights, no gate, no text. Every exogenous patch is a token
    (patch embedding + learned variable embedding + learned patch-position embedding); the target's patch embeddings
    are the queries; out = LayerNorm(q + Dropout(MHA(q, tokens, tokens))) -> flatten -> linear to the horizon, added to
    the backbone forecast. Width d_model = the side's magnet_d_model; the output layer starts at zero (training starts
    from plain DLinear, like MAGNet's cold start and ExoMLPDLinear).
    """

    def __init__(self, base, n_exo, patch_len, num_patches, horizon, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.base = base
        self.magnet_graph = None
        self.endo_emb = nn.Linear(patch_len, d_model)
        self.exo_emb = nn.Linear(patch_len, d_model)
        self.var_emb = nn.Parameter(torch.zeros(1, 1, n_exo, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches, 1, d_model))
        nn.init.normal_(self.var_emb, std=0.02); nn.init.normal_(self.pos_emb, std=0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(num_patches * d_model, horizon)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, exo_patches, endo_patches):
        out = self.base(exo_patches, endo_patches)
        squeeze = out.dim() == 3
        out = out.squeeze(-1) if squeeze else out
        B, P, _, n = exo_patches.shape
        q = self.endo_emb(endo_patches[..., 0]) + self.pos_emb[:, :, 0]                        # [B, P, d]
        kv = self.exo_emb(exo_patches.permute(0, 1, 3, 2)) + self.var_emb + self.pos_emb      # [B, P, n_exo, d]
        kv = kv.reshape(B, P * n, -1)
        h = self.norm(q + self.drop(self.attn(q, kv, kv, need_weights=False)[0]))
        out = out + self.head(h.flatten(1))
        return out.unsqueeze(-1) if squeeze else out
