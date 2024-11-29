import torch
import torch.nn as nn
import torch.nn.functional as F

from onpolicy.algorithms.utils.util import dict_merge
from onpolicy.algorithms.utils.attn import NoisySoftmaxAttention

def init(module, weight_init, bias_init, gain=1):
            weight_init(module.weight.data, gain=gain)
            if module.bias is not None:
                bias_init(module.bias.data)
            return module

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CommsTransformer(nn.Module):
    def __init__(self, dim, depth, heads, qk_dim, v_dim, mlp_dim, dropout):
        super(CommsTransformer, self).__init__()
        self.layers = nn.ModuleList([])
        self.bits_mat = 0
        
        self.fc = nn.Sequential(nn.Linear(dim, dim), nn.GELU())

        self.embedding = nn.Sequential(
			init_(nn.Linear(dim, dim*2, bias=True), activate=True),
			nn.GELU(),
			nn.LayerNorm(dim*2),
			init_(nn.Linear(dim*2, dim, bias=True), activate=True),
			nn.GELU(),
			nn.LayerNorm(dim),
		)
        
        # AttentionLayer = get_attention_layer(attention)
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, NoisySoftmaxAttention(dim=dim, qk_dim=qk_dim, v_dim=v_dim, heads=heads,
                                                           dropout=dropout),),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x, dist=False, mask=None):
        x = self.fc(x)
        batch, timesteps, n_agents, s_dim = x.shape
        x = x.reshape(-1, n_agents, s_dim)
        
        infos = []
        for attn, ff in self.layers:
            x_, info = attn(x, dist=dist, mask=mask)
            x = x_ + x
            x = ff(x) + x
            infos.append(info)

            if 'bits_mat' in info:
                self.bits_mat = info['bits_mat'].cpu().detach().numpy()

        info = dict_merge(infos, mode="mean")

        x = x.reshape(batch, timesteps, n_agents, s_dim)

        return x, info
    
class CommsMLP(nn.Module):
    def __init__(self, obs_input_dim) -> None:
        super(CommsMLP, self).__init__()
        
        self.fc = nn.Sequential(nn.Linear(obs_input_dim, obs_input_dim), nn.GELU())

        self.mlp = nn.Sequential(
            init_(nn.Linear(obs_input_dim, 2*obs_input_dim), activate=True),
			nn.GELU(),
			init_(nn.Linear(2*obs_input_dim, obs_input_dim), activate=True),
			nn.GELU()
        )

        self.embedding = nn.Sequential(
			init_(nn.Linear(obs_input_dim, obs_input_dim*2, bias=True), activate=True),
			nn.GELU(),
			nn.LayerNorm(obs_input_dim*2),
			init_(nn.Linear(obs_input_dim*2, obs_input_dim, bias=True), activate=True),
			nn.GELU(),
			nn.LayerNorm(obs_input_dim),
		)

    def forward(self, x, mask=None):
        x = self.fc(x)
        x = self.mlp(x)
        x = self.embedding(x)

        return x
