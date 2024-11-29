import torch
import torch.nn as nn
import torch.nn.functional as F

from onpolicy.algorithms.utils.comms import CommsTransformer, CommsMLP
from onpolicy.utils.util import get_shape_from_obs_space

class CommsNetwork:
    def __init__(self, args, obs_space, device=torch.device("cpu")):
        self.args = args
        self.obs_space = obs_space
        self.device = device

        obs_shape = get_shape_from_obs_space(obs_space)
        if type(obs_shape) == list:
            obs_shape = obs_shape[0]
        
        assert args.comms, "[ERROR]: Attempting to initialize CommsNetwork without comms enabled"
        assert args.comms_experiment != "NoComms", "Disable comms to run NoComms experiment "

        if args.comms_experiment == "FullComms":
            self.comms_net = CommsMLP(obs_shape)
            self.network_type = "MLP"

        elif args.comms_experiment == "LimitedComms" or args.comms_experiment == "Comms":
            self.comms_net = CommsTransformer(
                dim=obs_shape,
                depth=args.comms_depth,
                heads=args.comms_heads,
                qk_dim=args.comms_qk_dim,
                v_dim=args.comms_v_dim,
                mlp_dim=args.comms_mlp_dim,
                dropout=args.comms_dropout
            )
            self.network_type = "Transformer"

        self.info = {}
        self.lr = args.lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.comms_penalty = args.comms_penalty

        if args.comms_experiment == "LimitedComms" and self.comms_penalty == 0.0:
            print(f"[WARNING]: LimitedComms experiment with comms_penalty = 0.0")
        
        if args.comms_experiment == "Comms" and self.comms_penalty != 0.0:
            self.comms_penalty = 0.0
            print(f"[WARNING]: Forcibly setting comms_penalty to 0.0 for Comms experiment")
        
        self.optimizer = torch.optim.Adam(
            self.comms_net.parameters(), 
            lr=self.lr, 
            eps=self.opti_eps, 
            weight_decay=self.weight_decay
        )

        self.comms_net.to(self.device)

    def communicate(self, x, indiv_masks=None):
        if self.network_type == "MLP":
            return self.comms_net(x)
        
        elif self.network_type == "Transformer":
            x, self.info = self.comms_net(x, mask=indiv_masks)
            return x
        
    def get_comms_loss(self):
        comms_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_bits = 0.0
        for key, val in self.info.items():
            if "reg" in key:
                comms_loss = comms_loss + val * self.comms_penalty
                total_bits += val
        return comms_loss, total_bits
