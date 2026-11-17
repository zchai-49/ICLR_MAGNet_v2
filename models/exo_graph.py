# constructing exo-exo graphs
# identifying weights, node representation, and GAT network

import torch
import torch.nn as nn
import torch.nn.functional as F

class ExoEdgeWeightsFused(nn.Module):
    """
    Exo-exo edge weights fused with Rq text.
 
    Replaces ExoEdgeWeights in exo_graph.py when text is available.
 
    For each pair (i, j):
        numerical = cosine_sim(exo_i_emb, exo_j_emb)   scalar
        text      = rq_emb[exo_i_global, exo_j_global]  [d_model]
        fused     = MLP(concat(numerical, text))         scalar
 
    Then softmax over rows to get normalized attention weights.
 
    Args:
        n_exo      : number of exo variables
        d_model    : embedding dimension
        hidden_dim : MLP hidden size
 
    Input:
        exo_emb  : [B, n_exo, d_model]   exo patch embeddings
        rq_emb   : [n_vars, n_vars, d_model]  full relationship matrix
                   (n_vars = n_endo + n_exo, endo is index 0)
 
    Output:
        edge_weights : [B, n_exo, n_exo]
    """
 
    def __init__(self, n_exo: int, d_model: int, hidden_dim: int = 32):
        super().__init__()
        self.n_exo = n_exo
 
        # projection for cosine similarity (same as original ExoEdgeWeights)
        self.proj = nn.Linear(d_model, d_model)
 
        # shared RqEdgeFusion for all exo-exo pairs
        self.mlp = nn.Sequential(
            nn.Linear(1 + d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
 
    def forward(
        self,
        exo_emb : torch.Tensor,   # [B, n_exo, d_model]
        rq_emb  : torch.Tensor,   # [n_vars, n_vars, d_model]
    ) -> torch.Tensor:
 
        B, n_exo, d_model = exo_emb.shape
        if getattr(self, "uniform", False):      # rebuttal control: no cosine-similarity edge bias
            return exo_emb.new_full((B, n_exo, n_exo), 1.0 / n_exo)

    # cosine similarity
        x          = F.normalize(self.proj(exo_emb), p=2, dim=-1)
        cosine_sim = torch.bmm(x, x.transpose(-1, -2))   # [B, n_exo, n_exo]

    # rq_emb for exo-exo pairs: [n_exo, n_exo, d_model]
        if rq_emb is not None:
            rq_exo = rq_emb[1:, 1:]                           # [n_exo, n_exo, d_model]

    # cosine_sim: [B, n_exo, n_exo] → [B, n_exo, n_exo, 1]
            num = cosine_sim.unsqueeze(-1)                     # [B, n_exo, n_exo, 1]

    # rq_exo: [n_exo, n_exo, d_model] → [1, n_exo, n_exo, d_model]
            txt = rq_exo.unsqueeze(0).expand(B, -1, -1, -1)   # [B, n_exo, n_exo, d_model]

    # concat: [B, n_exo, n_exo, 1+d_model]
            combined = torch.cat([num, txt], dim=-1)

    # MLP on all pairs at once
            fused_weights = self.mlp(combined).squeeze(-1)     # [B, n_exo, n_exo]
        else:
            fused_weights = cosine_sim
        return F.softmax(fused_weights, dim=-1)
    
class ExoGAT(nn.Module):
    """
    Simplified Graph Attention Network for the exo-exo graph.
 
    Each node updates its embedding by:
      1. Gathering neighbor embeddings weighted by edge weights
      2. Transforming the aggregated result with a linear layer
      3. Adding back the original embedding (residual connection)
      4. Normalizing
 
    This is ONE GAT layer. Multiple layers = deeper message passing.
 
    Why residual connection?
      - Prevents the node from forgetting its own original embedding
      - Stabilizes training (same idea as ResNet)
 
    Args:
        d_model  : embedding dimension
        n_heads  : number of attention heads (multi-head for richer learning)
        dropout  : dropout for regularization
 
    Input:
        exo_emb      : [B, n_exo, d_model]   node embeddings before update
        edge_weights : [B, n_exo, n_exo]     from ExoEdgeWeights
 
    Output:
        updated_emb  : [B, n_exo, d_model]   node embeddings after update
    """
 
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
 
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads   # dimension per head
 
        # separate Q, K, V projections (standard attention)
        # Q = what am I looking for?
        # K = what do I have to offer?
        # V = what information do I actually pass?
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
 
        # output projection after gathering neighbor info
        self.W_out = nn.Linear(d_model, d_model)
 
        # layer norm + dropout for stability
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
 
    def forward(
        self,
        exo_emb     : torch.Tensor,   # [B, n_exo, d_model]
        edge_weights: torch.Tensor,   # [B, n_exo, n_exo]
    ) -> torch.Tensor:
 
        B, n_exo, d_model = exo_emb.shape
        residual = exo_emb                          # save for residual connection
 
        # ── project into Q, K, V ─────────────────────────────────────────────
        Q = self.W_q(exo_emb)                       # [B, n_exo, d_model]
        K = self.W_k(exo_emb)                       # [B, n_exo, d_model]
        V = self.W_v(exo_emb)                       # [B, n_exo, d_model]
 
        # ── split into multiple heads ─────────────────────────────────────────
        # reshape: [B, n_exo, d_model] → [B, n_heads, n_exo, d_head]
        def split_heads(x):
            return x.reshape(B, n_exo, self.n_heads, self.d_head) \
                    .permute(0, 2, 1, 3)            # [B, n_heads, n_exo, d_head]
 
        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)
 
        # ── compute attention scores ──────────────────────────────────────────
        # raw attention: how much does node i want to attend to node j?
        scale   = self.d_head ** 0.5
        attn    = torch.matmul(Q, K.transpose(-1, -2)) / scale
        #         [B, n_heads, n_exo, n_exo]
 
        # ── blend with edge weights from Step 1 ──────────────────────────────
        # edge_weights encode structural similarity (cosine sim)
        # attn encodes content-based similarity (Q·K)
        # combining both gives richer, more informed attention
        edge_w  = edge_weights.unsqueeze(1)         # [B, 1, n_exo, n_exo] → broadcasts over heads
        attn    = F.softmax(attn + edge_w.log().clamp(min=-1e9), dim=-1)
        #         [B, n_heads, n_exo, n_exo]
 
        # ── gather neighbor information ───────────────────────────────────────
        # each node collects a weighted sum of its neighbors' V vectors
        out     = torch.matmul(attn, V)             # [B, n_heads, n_exo, d_head]
 
        # ── merge heads back ──────────────────────────────────────────────────
        out     = out.permute(0, 2, 1, 3)           # [B, n_exo, n_heads, d_head]
        out     = out.reshape(B, n_exo, d_model)    # [B, n_exo, d_model]
 
        # ── output projection + residual + norm ───────────────────────────────
        out     = self.W_out(out)                   # [B, n_exo, d_model]
        out     = self.norm(residual + self.dropout(out))
 
        return out
    
class VqCrossAttention(nn.Module):
    """
    Cross-attention over a REAL multi-token text sequence.

    Previously this received one pooled [CLS] vector as its only key, which
    made softmax over a single key always evaluate to 1.0 — i.e. attention
    degenerated into a fixed scalar gate on one vector, with nothing to
    select between. Now the key/value sequence is the variable's full
    per-token BERT embedding (optionally padded, masked out via vq_mask),
    plus an optional extra "stats" token carrying the dynamic per-patch
    statistics — so the query can genuinely choose which sub-fact of the
    text (or the current stats) is relevant, instead of always drinking
    from a single fixed cup.
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.d_model = d_model

        self.W_q    = nn.Linear(d_model, d_model)
        self.W_k    = nn.Linear(d_model, d_model)
        self.W_v    = nn.Linear(d_model, d_model)
        self.W_out  = nn.Linear(d_model, d_model)
        self.norm   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        patch_emb : torch.Tensor,             # [N, d_model]  N = B*P (flattened)
        vq_tokens : torch.Tensor,              # [T, d_model]  static per-variable text tokens (already projected), possibly padded
        vq_mask   : torch.Tensor = None,       # [T]           True = real token, False = pad
        stats_emb : torch.Tensor = None,       # [N, d_model]  optional dynamic per-patch stats, appended as one extra token
    ) -> torch.Tensor:

        N        = patch_emb.shape[0]
        T        = vq_tokens.shape[0]
        residual = patch_emb                           # save for residual

        # patch_emb as Query: [N, 1, d_model]
        Q = self.W_q(patch_emb).unsqueeze(1)

        # static text tokens, broadcast over N: [N, T, d_model]
        vq = vq_tokens.unsqueeze(0).expand(N, T, self.d_model)
        mask = vq_mask.unsqueeze(0).expand(N, T) if vq_mask is not None else None

        # dynamic per-patch stats as one extra key/value token
        if stats_emb is not None:
            stats_tok = stats_emb.unsqueeze(1)                       # [N, 1, d_model]
            kv_in     = torch.cat([vq, stats_tok], dim=1)            # [N, T+1, d_model]
            if mask is not None:
                extra = torch.ones(N, 1, dtype=torch.bool, device=mask.device)
                mask  = torch.cat([mask, extra], dim=1)              # [N, T+1]
        else:
            kv_in = vq

        K  = self.W_k(kv_in)                          # [N, T(+1), d_model]
        V  = self.W_v(kv_in)
        T_kv = K.shape[1]

        # split into heads
        def split_heads(x, seq_len):
            return x.reshape(N, seq_len, self.n_heads, self.d_head) \
                    .permute(0, 2, 1, 3)              # [N, n_heads, seq_len, d_head]

        Q = split_heads(Q, 1)
        K = split_heads(K, T_kv)
        V = split_heads(V, T_kv)

        # scaled dot-product attention over the real multi-token sequence
        scale = self.d_head ** 0.5
        attn  = torch.matmul(Q, K.transpose(-1, -2)) / scale
        #       [N, n_heads, 1, T_kv]

        if mask is not None:
            # mask out padded key positions before softmax
            mask_bias = mask.unsqueeze(1).unsqueeze(1)     # [N, 1, 1, T_kv]
            attn = attn.masked_fill(~mask_bias, float('-inf'))

        attn  = F.softmax(attn, dim=-1)               # [N, n_heads, 1, T_kv]

        # weighted value
        out   = torch.matmul(attn, V)                 # [N, n_heads, 1, d_head]

        # merge heads
        out   = out.permute(0, 2, 1, 3)               # [N, 1, n_heads, d_head]
        out   = out.reshape(N, self.d_model)          # [N, d_model]

        # output projection + residual + norm
        out   = self.W_out(out)                       # [N, d_model]
        out   = self.norm(residual + self.dropout(out))

        return out                                    # [N, d_model]
    
class ExoNodeConstruction(nn.Module):
    """
    Applies Vq cross-attention to all exo and endo patch embeddings.
 
    Uses SEPARATE cross-attention modules for exo and endo nodes
    since they have different roles in the graph.
 
    Processes all variables and all patches efficiently by looping
    over the patch and variable dimensions.
 
    Args:
        n_exo   : number of exogenous variables
        d_model : embedding dimension
        n_heads : attention heads
        dropout : dropout rate
 
    Input:
        exo_emb_ts  : [B, num_patches, n_exo, d_model]  from TSPreprocessor
        endo_emb_ts : [B, num_patches, 1,     d_model]  from TSPreprocessor
        vq_emb      : [n_vars, d_model]                 precomputed text embeddings
                      ordered as [endo_0, exo_0, exo_1, ...]
                      matching DATASET_INFO['all_vars'] order
 
    Output:
        exo_fused  : [B, num_patches, n_exo, d_model]
        endo_fused : [B, num_patches, 1,     d_model]
    """
 
    def __init__(
        self,
        n_exo   : int,
        d_model : int,
        n_heads : int   = 4,
        dropout : float = 0.1,
    ):
        super().__init__()
 
        # separate cross-attention for exo and endo
        self.exo_cross_attn  = VqCrossAttention(d_model, n_heads, dropout)
 
    def forward(self, exo_emb_ts, vq_tokens, vq_mask=None, stats_emb=None):
        # vq_tokens: [n_exo, T, d_model]  static per-token text (already projected), possibly padded
        # vq_mask  : [n_exo, T]           True = real token, False = pad (optional)
        # stats_emb: [B, P, d_model]      dynamic per patch, shared across variables
        B, P, n_exo, d_model = exo_emb_ts.shape
        exo_flat = exo_emb_ts.reshape(B * P, n_exo, d_model)

        # stats: [B, P, d_model] → [B*P, d_model]
        if stats_emb is not None:
            stats_flat = stats_emb.reshape(B * P, d_model)  # [B*P, d_model]

        var_fused = []
        for v in range(n_exo):
            patch = exo_flat[:, v, :]                        # [B*P, d_model]
            stats_v = stats_flat if stats_emb is not None else None

            if vq_tokens is None:
                # no-text ablation: skip cross-attention if there's also no
                # stats to inject, otherwise fall back to stats-only
                if stats_v is None:
                    var_fused.append(patch)
                    continue
                fused = self.exo_cross_attn(
                    patch,
                    torch.zeros(1, d_model, device=patch.device, dtype=patch.dtype),
                    stats_emb=stats_v,
                )
                var_fused.append(fused)
                continue

            vq_v   = vq_tokens[v]                            # [T, d_model]
            mask_v = vq_mask[v] if vq_mask is not None else None
            fused  = self.exo_cross_attn(patch, vq_v, vq_mask=mask_v, stats_emb=stats_v)
            var_fused.append(fused)

        exo_fused = torch.stack(var_fused, dim=1)
        return exo_fused.reshape(B, P, n_exo, d_model)

    
class ExoGraph(nn.Module):
    """
    Full exo-exo graph module for ONE patch.
 
    Combines:
        ExoEdgeWeights → edge weights  [B, n_exo, n_exo]
        ExoGAT         → updated node embeddings [B, n_exo, d_model]
 
    To process all patches, call this in a loop over the patch dimension
    (handled in the top-level MAGNET model later).
 
    Args:
        n_exo   : number of exogenous variables
        d_model : embedding dimension
        n_heads : GAT attention heads
        dropout : dropout rate
 
    Input:
        exo_emb : [B, n_exo, d_model]   exo embeddings for ONE patch
        rq_emb
 
    Output:
        updated_exo_emb : [B, n_exo, d_model]
        edge_weights    : [B, n_exo, n_exo]   (returned for endo insertion later)
    """
 
    def __init__(
        self,
        n_exo   : int,
        d_model : int,
        n_heads : int   = 4,
        dropout : float = 0.1,
    ):
        super().__init__()
        self.edge_weights = ExoEdgeWeightsFused(n_exo, d_model)
        self.gat          = ExoGAT(d_model, n_heads, dropout)
 
    def forward(self, exo_emb: torch.Tensor, rq_emb):
        # exo_emb: [B, n_exo, d_model]
        edge_w          = self.edge_weights(exo_emb, rq_emb)    # [B, n_exo, n_exo]
        updated_exo_emb = self.gat(exo_emb, edge_w)    # [B, n_exo, d_model]
        return updated_exo_emb, edge_w

if __name__ == '__main__':
    B           = 16    # batch size
    num_patches = 8     # number of patches per sample
    n_exo       = 4     # occupancy, speed
    n_vars      = 5
    d_model     = 64
    n_heads     = 4
 
    exo_graph = ExoGraph(n_exo=n_exo, d_model=d_model, n_heads=n_heads)
    node_construction = ExoNodeConstruction(
        n_exo   = n_exo,
        d_model = d_model,
        n_heads = n_heads,
    )
 
    # simulate enriched exo embeddings from TSPreprocessor
    # shape: [B, num_patches, n_exo, d_model]
    exo_emb_all = torch.randn(B, num_patches, n_exo, d_model)
    vq_emb = torch.randn(n_vars, d_model)
    rq_emb     = torch.randn(n_vars, n_vars, d_model)
 
    # process patch by patch (this loop will later live in the MAGNET model)
    updated_patches = []
    edge_weight_patches = []
 
    exo_fused = node_construction(exo_emb_all, vq_emb)
    for p in range(num_patches):
        exo_patch = exo_fused[:, p, :, :]            # [B, n_exo, d_model]
        updated, edge_w = exo_graph(exo_patch, rq_emb)
        updated_patches.append(updated)
        edge_weight_patches.append(edge_w)
 
    # stack back into [B, num_patches, n_exo, d_model]
    updated_exo_all  = torch.stack(updated_patches,      dim=1)
    edge_weights_all = torch.stack(edge_weight_patches,  dim=1)
 
    print("── ExoEdgeWeightsWithRq ──────────────────────────────")
    exo_edge = ExoEdgeWeightsFused(n_exo=n_exo, d_model=d_model)
    edge_w   = exo_edge(exo_emb_all[:, 0, :, :], rq_emb)
    print(f"  input  exo_emb    : {exo_emb_all.shape}")     # [16, 2, 64]
    print(f"  output edge_w     : {edge_w.shape}")       # [16, 2, 2]
    print(f"  sample 0 weights  :")
    print(f"              occ       spd")
    print(f"  occ   {edge_w[0,0,0]:.4f}    {edge_w[0,0,1]:.4f}")
    print(f"  spd   {edge_w[0,1,0]:.4f}    {edge_w[0,1,1]:.4f}")
    print(f"  row sums (=1.0)   : {edge_w[0].sum(dim=-1)}")
    print("── Exo-Exo Graph output shapes ───────────────────")
    print(f"  input  exo_emb      : {exo_emb_all.shape}")      # [16, 8, 2, 64]
    print(f"  output updated_exo  : {updated_exo_all.shape}")  # [16, 8, 2, 64]
    print(f"  output edge_weights : {edge_weights_all.shape}") # [16, 8, 2, 2]
    print()
    print("── Edge weights for patch 0, sample 0 ────────────")
    w = edge_weights_all[0, 0]
    print(f"              occupancy    speed")
    print(f"  occupancy   {w[0,0]:.4f}      {w[0,1]:.4f}")
    print(f"  speed       {w[1,0]:.4f}      {w[1,1]:.4f}")
    print()
    print("── Parameter count ───────────────────────────────")
    total = sum(p.numel() for p in exo_graph.parameters())
    print(f"  ExoGraph: {total:,} parameters")
    print()
    for name, module in exo_graph.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {params:,} parameters")
