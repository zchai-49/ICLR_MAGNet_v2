# text_projection.py
# Extracted from the original text_generate.py (LLM/BERT text-generation
# pipeline, not needed here -- text embeddings are precomputed and shipped
# under data/text_emb/). This is the one piece MAGNET_graph_only.py actually
# needs: projecting a precomputed BERT embedding from 768 -> d_model.

import torch


class TextProjection(torch.nn.Module):
    """
    Projects BERT [CLS] embeddings from 768 -> d_model.
    Trained jointly with MAGNet (not frozen).
    """
    def __init__(self, bert_dim: int = 768, d_model: int = 64):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(bert_dim, d_model * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
