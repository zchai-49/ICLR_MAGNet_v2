import torch
import torch.nn as nn
import torch.nn.functional as F

from timexer.layers.Embed import DataEmbedding_inverted, PositionalEmbedding
from timexer.layers.SelfAttention_Family import FullAttention, AttentionLayer


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars  = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear  = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs, nvars, d_model, patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.value_embedding   = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token         = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout           = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        glb    = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers     = nn.ModuleList(layers)
        self.norm       = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask,
                      tau=tau, delta=delta)
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention  = self_attention
        self.cross_attention = cross_attention
        self.conv1  = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2  = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)
        self.norm3  = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape

        x = x + self.dropout(self.self_attention(
            x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0])
        x = self.norm1(x)

        x_glb_ori  = x[:, -1, :].unsqueeze(1)
        x_glb      = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta)[0])
        x_glb_attn = torch.reshape(
            x_glb_attn,
            (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])
        ).unsqueeze(1)
        x_glb = self.norm2(x_glb_ori + x_glb_attn)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class timexer_model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.seq_len    = configs.seq_len
        self.pred_len   = configs.pred_len
        self.use_norm   = True
        self.patch_len  = configs.patch_len
        self.patch_num  = int(configs.seq_len // configs.patch_len)
        self.n_vars     = configs.enc_in
        self.series_dim = configs.series_dim
        self.use_future = configs.use_future

        # ── Embeddings ──────────────────────────────────────────────────────
        self.en_embedding = EnEmbedding(
            self.series_dim, configs.d_model, self.patch_len, configs.dropout
        )
        seq_exo = configs.seq_len + configs.pred_len if self.use_future else configs.seq_len
        self.ex_embedding = DataEmbedding_inverted(
            seq_exo, configs.d_model, None, configs.freq, configs.dropout
        )

        # ── Encoder ─────────────────────────────────────────────────────────
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # ── Heads ────────────────────────────────────────────────────────────
        self.head_nf = configs.d_model * (self.patch_num + 1)

        # Vanilla TimeXer head (used in forecast / forward_only)
        self.head = FlattenHead(
            configs.enc_in, self.head_nf, configs.pred_len,
            head_dropout=configs.dropout
        )

        # MAGNET hybrid head — only 1 variable (endo)
        self.magnet_head = FlattenHead(
            n_vars=1, nf=self.head_nf, target_window=configs.pred_len,
            head_dropout=configs.dropout
        )

        # Global token for MAGNET hybrid path (properly registered)
        self.magnet_glb_token = nn.Parameter(torch.randn(1, 1, configs.d_model))

        # Exo projection for MAGNET hybrid path
        self.magnet_exo_proj = nn.Linear(
            configs.seq_len + configs.pred_len, configs.d_model
        )

    # ── Vanilla TimeXer path ─────────────────────────────────────────────────

    def forecast(self, x_history, exog):
        """Core TimeXer forward — used by forward_only (mode 3)."""
        en_embed, n_vars = self.en_embedding(x_history.permute(0, 2, 1))
        ex_embed = self.ex_embedding(exog, None)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        enc_out = enc_out.permute(0, 1, 3, 2)      # [B, n_vars, d_model, P+1]
        dec_out = self.head(enc_out)                 # [B, n_vars, pred_len]
        dec_out = dec_out.permute(0, 2, 1)           # [B, pred_len, n_vars]
        return dec_out

    def forward_only(self, x_enc, exog_future):
        """Mode 3 entry point — vanilla TimeXer."""
        dec_out = self.forecast(x_enc, exog_future)
        return dec_out[:, -self.pred_len:, :]        # [B, pred_len, n_vars]

    # ── MAGNET hybrid path ───────────────────────────────────────────────────

    def forward_with_magnet(self, endo_history, full_exog, graph_bias=None):
        """
        Mode 2 entry point — TimeXer with MAGNET causal graph bias.

        Args:
            endo_history : [B, seq_len, 1]        raw endo sequence
            full_exog    : [B, seq_len+pred_len, n_exo]  full exo (history + future)
            graph_bias   : [B, P, d_model]         causal bias from MAGNETGraphOnly
                           (None = no injection, behaves like vanilla TimeXer)
        """
        B = endo_history.shape[0]

        # 1. TimeXer's own patch embedding on raw endo sequence
        en_embed, n_vars = self.en_embedding(endo_history.permute(0, 2, 1))
        # en_embed: [B*1, P+1, d_model]

        # 2. Inject MAGNET graph bias into endo patch embeddings (not the glb token)
        if graph_bias is not None:
            # en_embed: [B, P+1, d_model] (n_vars=1 so B*1=B)
            # graph_bias: [B, P, d_model]
            # add to patch positions only, not the global token (last position)
            en_embed[:, :-1, :] = en_embed[:, :-1, :] + graph_bias

        # 3. Exo cross-attention embedding
        ex_embed = self.ex_embedding(full_exog, None)  # [B, n_exo, d_model]

        # 4. Transformer encoder
        enc_out = self.encoder(en_embed, ex_embed)     # [B, P+1, d_model]

        # 5. Head: reshape for FlattenHead [B, 1, d_model, P+1]
        enc_out = enc_out.unsqueeze(1).permute(0, 1, 3, 2)
        dec_out = self.magnet_head(enc_out)            # [B, 1, pred_len]
        dec_out = dec_out.permute(0, 2, 1).squeeze(-1) # [B, pred_len]

        return dec_out

    # ── Helpers ──────────────────────────────────────────────────────────────

    def sample_norm(self, x, means, stdev):
        return (x - means) / stdev

    def sample_denorm(self, x, means, stdev):
        seq_len = x.shape[1]
        x = x * stdev[:, 0, :].unsqueeze(1).repeat(1, seq_len, 1)
        x = x + means[:, 0, :].unsqueeze(1).repeat(1, seq_len, 1)
        return x