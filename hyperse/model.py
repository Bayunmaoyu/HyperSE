import torch
from torch import nn


def mlp(inp, hidden, out, dropout):
    return nn.Sequential(nn.Linear(inp,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,out))


class HyperSE(nn.Module):
    def __init__(self, node_dim, dual_dim, stats_dim=18, hidden=64, dropout=0.1,
                 mode='enhanced', ablation='none'):
        super().__init__()
        self.mode, self.ablation = mode, ablation
        self.node_mlp = mlp(node_dim,hidden,hidden,dropout)
        self.hyper_mlp = mlp(hidden,hidden,hidden,dropout)
        self.dual_mlp = mlp(dual_dim,hidden,hidden,dropout)
        self.decoder = mlp(2*hidden,hidden,1,dropout)
        if mode == 'enhanced':
            self.stats_mlp = mlp(stats_dim,hidden,1,dropout)
            self.stats_linear = nn.Linear(stats_dim,1)

    def forward(self, node_features, node_mask, dual_features, neighbor_mask, stats):
        # Eq. (16): pointwise MLP BEFORE masked sum, then f.
        hyper = self.hyper_mlp((self.node_mlp(node_features)*node_mask.unsqueeze(-1)).sum(1))
        # Eq. (23): unweighted average of selected neighbor MLPs.
        # If fewer than l exist, divide by actual count. Empty neighborhood -> 0.
        dual = (self.dual_mlp(dual_features)*neighbor_mask.unsqueeze(-1)).sum(1)
        dual = dual/neighbor_mask.sum(1,keepdim=True).clamp_min(1)
        if self.ablation == 'no_primal':
            hyper = torch.zeros_like(hyper)
        elif self.ablation == 'no_dual':
            dual = torch.zeros_like(dual)
        score = self.decoder(torch.cat([hyper,dual],dim=-1)).squeeze(-1)
        if self.mode == 'enhanced':
            # Optional history-only residual; completely absent in core mode.
            score = score + self.stats_mlp(stats).squeeze(-1) + self.stats_linear(stats).squeeze(-1)
        return score  # logits; BCEWithLogitsLoss, sigmoid only for reporting
