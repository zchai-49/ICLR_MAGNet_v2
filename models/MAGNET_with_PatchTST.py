# MAGNET_with_PatchTST.py
# Mode 2: MAGNET causal graph bias injected into PatchTST patch embeddings
# Mode 3: Vanilla PatchTST only (no MAGNET)
#
# Key advantage over TimeXer integration:
#   PatchTST is channel-independent and uses NO exo cross-attention.
#   MAGNET's graph bias is the ONLY exo signal — clean test of causal graph value.

import math
import torch
import torch.nn as nn

from MAGNET_graph_only import MAGNETGraphOnly


# ── Reused from PatchTST (self-contained) ─────────────

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear  = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # [B, n_vars, d_model, patch_num]
        x = self.flatten(x)   # [B, n_vars, d_model * patch_num]
        x = self.linear(x)    # [B, n_vars, target_window]
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """
    Simple patch embedding: unfold → linear projection → positional encoding.
    Mirrors PatchTST's PatchEmbedding but self-contained.
    """
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        self.padding   = padding
        self.proj      = nn.Linear(patch_len, d_model)
        self.norm      = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):  # x: [B*n_vars, seq_len]  (already permuted)
        # pad end
        x = nn.functional.pad(x, (0, self.padding))
        # unfold into patches: [B*n_vars, num_patches, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # project: [B*n_vars, num_patches, d_model]
        x = self.dropout(self.norm(self.proj(x)))
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.self_attn(x, x, x, need_weights=False)[0])
        x = self.norm2(x + self.ff(x))
        return x


class PatchTSTBackbone(nn.Module):
    """
    PatchTST backbone: patch embedding → transformer encoder → flatten head.
    Channel-independent (processes each variable separately).
    Accepts optional graph_bias [B, num_patches, d_model] for MAGNET injection.
    """
    def __init__(
        self,
        seq_len        : int,
        pred_len       : int,
        patch_len      : int,
        stride         : int,
        d_model        : int = 128,
        n_heads        : int = 8,
        e_layers       : int = 3,
        d_ff           : int = 256,
        dropout        : float = 0.2,
        head_dropout   : float = 0.0,
        n_vars         : int = 1,
    ):
        super().__init__()
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.n_vars   = n_vars
        padding = stride

        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, padding, dropout)

        # num_patches = floor((seq_len - patch_len) / stride) + 2  (with padding)
        self.num_patches = int((seq_len - patch_len) / stride + 2)

        self.encoder = nn.Sequential(*[
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(e_layers)
        ])

        head_nf = d_model * self.num_patches
        self.head = FlattenHead(n_vars, head_nf, pred_len, head_dropout)

    def forward(self, x, graph_bias=None):
        """
        x           : [B, seq_len, n_vars]  (raw, already normalized)
        graph_bias  : [B, num_patches, d_model] or None
        returns     : [B, pred_len, n_vars]
        """
        B, T, C = x.shape

        # channel-independent: merge batch and var dims
        x = x.permute(0, 2, 1)              # [B, n_vars, seq_len]
        x = x.reshape(B * C, T)             # [B*n_vars, seq_len]

        # patch embedding
        enc = self.patch_embedding(x)        # [B*n_vars, num_patches, d_model]

        # inject graph bias into endo variable (var index 0) only
        if graph_bias is not None:
            # graph_bias: [B, num_patches, d_model]
            # endo is first variable in channel-independent layout
            # enc[0::C] = endo samples (every C-th row starting at 0)
            P_bias = min(graph_bias.shape[1], enc.shape[1])
            enc_endo = enc.reshape(B, C, self.num_patches, -1)[:, 0, :, :]  # [B, P, d]
            enc_endo[:, :P_bias, :] = enc_endo[:, :P_bias, :] + graph_bias[:, :P_bias, :]
            enc_reshaped = enc.reshape(B, C, self.num_patches, -1)
            enc_reshaped[:, 0, :, :] = enc_endo
            enc = enc_reshaped.reshape(B * C, self.num_patches, -1)

        # transformer encoder
        enc = self.encoder(enc)              # [B*n_vars, num_patches, d_model]

        # reshape for head: [B, n_vars, d_model, num_patches]
        enc = enc.reshape(B, C, self.num_patches, -1)
        enc = enc.permute(0, 1, 3, 2)       # [B, n_vars, d_model, num_patches]

        # head → [B, n_vars, pred_len] → [B, pred_len, n_vars]
        out = self.head(enc).permute(0, 2, 1)
        return out


# ── Main wrapper ──────────────────────────────────────────────────────────────

class MAGNETWithPatchTST(nn.Module):
    """
    Hybrid model: PatchTST backbone + optional MAGNET causal graph bias.

    Mode 2 — MAGNET graph bias injected into PatchTST patch embeddings:
        1. PatchTST handles its own normalization + patch embedding on raw endo
        2. MAGNETGraphOnly produces causal bias [B, P, d_graph]
        3. Bias projected to PatchTST d_model, added to endo patch embeddings
        4. PatchTST encoder + head runs as normal

    Mode 3 — Vanilla PatchTST only:
        Pure PatchTST on endo series, no MAGNET involvement.

    Key difference vs TimeXer integration:
        PatchTST has NO exo cross-attention — MAGNET's graph is the only
        source of exo information, giving a clean test of causal graph value.
    """

    def __init__(self, magnet_config: dict, patchtst_config: dict, mode: int = 2,
                 use_text: bool = True):
        super().__init__()
        self.mode = mode

        seq_len  = patchtst_config['seq_len']
        pred_len = patchtst_config['pred_len']
        patch_len = patchtst_config['patch_len']
        stride   = patchtst_config.get('stride', patch_len // 2)
        d_model  = patchtst_config.get('d_model', 128)
        n_heads  = patchtst_config.get('n_heads', 8)
        e_layers = patchtst_config.get('e_layers', 3)
        d_ff     = patchtst_config.get('d_ff', 256)
        dropout  = patchtst_config.get('dropout', 0.2)

        # ── MAGNET graph module (mode 2 only) ─────────────────────────────
        if self.mode == 2:
            self.magnet_graph = MAGNETGraphOnly(
                n_exo          = magnet_config['n_exo'],
                patch_len      = magnet_config['patch_len'],
                d_model        = magnet_config.get('d_model', 64),
                use_text       = use_text,
                n_heads        = magnet_config.get('n_heads', 4),
                ngc_hidden_dim = magnet_config.get('ngc_hidden_dim', 32),
                dropout        = magnet_config.get('dropout', 0.1),
                dataset        = magnet_config['dataset'],
                term           = magnet_config['term'],
                ngc_lag        = magnet_config.get('ngc_lag', 1),
            )
            # project graph d_model → PatchTST d_model
            self.graph_proj = nn.Sequential(
                nn.Linear(magnet_config.get('d_model', 64), d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(0.1),
            )
        else:
            self.magnet_graph = None
            self.graph_proj   = None

        # ── PatchTST backbone ─────────────────────────────────────────────
        self.backbone = PatchTSTBackbone(
            seq_len      = seq_len,
            pred_len     = pred_len,
            patch_len    = patch_len,
            stride       = stride,
            d_model      = d_model,
            n_heads      = n_heads,
            e_layers     = e_layers,
            d_ff         = d_ff,
            dropout      = dropout,
            n_vars       = 1,               # channel-independent, endo only
        )

    def forward(
        self,
        exo_patches   : torch.Tensor,         # [B, P, patch_len, n_exo]
        endo_patches  : torch.Tensor,         # [B, P, patch_len, 1]
        exog_future   : torch.Tensor = None,  # unused by PatchTST, kept for API compat
        time_features : torch.Tensor = None,  # unused by PatchTST
        dataset       : int = None,
    ) -> torch.Tensor:
        if exog_future is not None:
            raise ValueError("future exogenous values are not accepted: forecasts use the observed history only")

        B, P, pl, _ = endo_patches.shape

        # reconstruct raw endo sequence
        endo_history = endo_patches.reshape(B, P * pl, 1)  # [B, seq_len, 1]

        # instance normalization
        means = endo_history.mean(dim=1, keepdim=True).detach()   # [B, 1, 1]
        stdev = torch.sqrt(
            torch.var(endo_history, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()                                                  # [B, 1, 1]
        endo_norm = (endo_history - means) / stdev                 # [B, seq_len, 1]

        # scalar mean/std for denorm
        m = means[:, 0, 0].unsqueeze(1)   # [B, 1]
        s = stdev[:, 0, 0].unsqueeze(1)   # [B, 1]

        if self.mode == 3:
            out = self.backbone(endo_norm).squeeze(-1)  # [B, pred_len]
            return out * s + m

        graph_bias = self.magnet_graph(exo_patches, endo_patches)
        graph_bias = self.graph_proj(graph_bias)
        out = self.backbone(endo_norm, graph_bias=graph_bias).squeeze(-1)  # [B, pred_len]
        return out * s + m