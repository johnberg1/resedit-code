import torch.nn as nn

class AdvMLP(nn.Module):
    def __init__(self, d_prompt, d_target, hidden=1024, depth=2, drop=0.1):
        super().__init__()
        layers = [nn.Linear(d_prompt, hidden), nn.GELU(), nn.Dropout(drop)]
        for _ in range(depth-1):
            layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop)]
        layers += [nn.Linear(hidden, d_target)]
        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, hidden_states):
        return self.net(hidden_states)
