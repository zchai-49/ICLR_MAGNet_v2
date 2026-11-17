# MAGNET_with_DAG.py
# Mode 2: MAGNet causal graph bias injected into DAG's input before both of its encoders
# Mode 3: Vanilla DAG only
#
# DAG operates on raw sequences [B, seq_len, 1 + n_exo] (endogenous channel first). MAGNet's graph produces a
# patch-level causal bias [B, P, d_graph]; it is projected to one number per patch, upsampled to the sequence
# length, scaled by a learned sigmoid gate (initialised at sigmoid(-2) so it starts mostly closed) and added to the
# ENDOGENOUS channel only, before DAG's temporal and channel-covariance encoders.
#
# Future exogenous values: DAG's own training objective uses the true future exogenous series as a teacher
# (an auxiliary causality loss and teacher forcing in its encoders). This wrapper therefore accepts `exog_future`
# ONLY in training mode. In eval mode (validation, test) it raises if given one, and DAG runs with
# infer_use_future=False, so the model never sees future exogenous values at inference.

import torch
import torch.nn as nn

from MAGNET_graph_only import MAGNETGraphOnly


class MAGNETWithDAG(nn.Module):
    def __init__(self, magnet_config: dict, dag_config, mode: int = 2,
                 use_text: bool = True, text_variant: str = "new",
                 node_text_fusion: str = "qk_bias",
                 use_instance_norm: bool = True):
        super().__init__()
        if mode not in (2, 3):
            raise ValueError("mode must be 2 (DAG + MAGNet) or 3 (vanilla DAG)")
        if getattr(dag_config, "infer_use_future", False):
            raise ValueError("dag_config.infer_use_future must be False: no future exogenous values at inference")
        self.mode = mode
        self.use_instance_norm = use_instance_norm
        self.seq_len = dag_config.seq_len

        if self.mode == 2:
            d_graph = magnet_config.get('d_model', 64)
            self.magnet_graph = MAGNETGraphOnly(
                n_exo          = magnet_config['n_exo'],
                patch_len      = magnet_config['patch_len'],
                d_model        = d_graph,
                n_heads        = magnet_config.get('n_heads', 4),
                ngc_hidden_dim = magnet_config.get('ngc_hidden_dim', 32),
                dropout        = magnet_config.get('dropout', 0.1),
                dataset        = magnet_config['dataset'],
                term           = magnet_config['term'],
                use_text       = use_text,
                text_variant   = text_variant,
                node_text_fusion = node_text_fusion,
                ngc_lag        = magnet_config.get('ngc_lag', 1),
            )
            # graph bias [B, P, d_graph] -> [B, P, 1] -> upsample -> [B, seq_len, 1]
            self.graph_proj = nn.Sequential(nn.Linear(d_graph, 16), nn.GELU(), nn.Linear(16, 1))
            self.bias_scale = nn.Parameter(torch.tensor(-2.0))
        else:
            self.magnet_graph = None
            self.graph_proj = None

        from ts_benchmark.baselines.dag.models.dag_model import DAGModel
        self.dag = DAGModel(dag_config)

    def forward(
        self,
        exo_patches   : torch.Tensor,         # [B, P, patch_len, n_exo]
        endo_patches  : torch.Tensor,         # [B, P, patch_len, 1]
        exog_future   : torch.Tensor = None,  # [B, pred_len, n_exo]; training-time teacher only
        time_features : torch.Tensor = None,  # unused
        dataset       : int = None,
    ) -> torch.Tensor:
        if exog_future is not None and not self.training:
            raise ValueError("future exogenous values are only accepted in training mode (DAG's teacher signal); "
                             "evaluation uses the observed history only")

        B, P, pl, n_exo = exo_patches.shape
        endo_seq = endo_patches.reshape(B, P * pl, 1)
        exo_seq  = exo_patches.reshape(B, P * pl, n_exo)
        dag_input = torch.cat([endo_seq, exo_seq], dim=-1)      # [B, seq_len, 1+n_exo]

        if self.use_instance_norm:                              # RevIN-style, per window and channel
            means = dag_input.mean(dim=1, keepdim=True).detach()
            stdev = torch.sqrt(torch.var(dag_input, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            dag_input = (dag_input - means) / stdev
        else:
            means = torch.zeros(dag_input.shape[0], 1, dag_input.shape[2], device=dag_input.device, dtype=dag_input.dtype)
            stdev = torch.ones_like(means)

        if exog_future is not None:                             # training only
            if len(exog_future.shape) == 4:
                exog_future = exog_future.reshape(B, -1, n_exo)
            dag_future = (exog_future - means[:, :, 1:]) / stdev[:, :, 1:]   # same history-derived stats
        else:
            dag_future = None

        endo_m = means[:, 0, 0].unsqueeze(1)
        endo_s = stdev[:, 0, 0].unsqueeze(1)

        if self.mode == 3:
            output, causality_loss = self.dag(dag_input, dag_future)
            pred = output[:, :, 0] * endo_s + endo_m
            return (pred, causality_loss) if self.training else pred

        graph_bias = self.magnet_graph(exo_patches, endo_patches)          # [B, P, d_graph]
        graph_bias_1d = self.graph_proj(graph_bias)                        # [B, P, 1]
        graph_bias_seq = nn.functional.interpolate(
            graph_bias_1d.permute(0, 2, 1), size=self.seq_len, mode='linear', align_corners=False,
        ).permute(0, 2, 1)                                                 # [B, seq_len, 1]
        causal_bias = torch.sigmoid(self.bias_scale) * graph_bias_seq

        dag_input_enriched = dag_input.clone()
        dag_input_enriched[:, :, 0:1] = dag_input[:, :, 0:1] + causal_bias

        output, causality_loss = self.dag(dag_input_enriched, dag_future)
        pred = output[:, :, 0] * endo_s + endo_m
        return (pred, causality_loss) if self.training else pred
