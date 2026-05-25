import torch
import torch.nn as nn

class William(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout=0, bidirectional=1, temperature=1):
        super(William, self).__init__()
        
        self.D = max(1, min(bidirectional, 2))
        self.H_c = hidden_dim
        self.H_i = input_dim 
        self.H_o = output_dim 
        self.N_LAYERS = layer_dim
        self.T = temperature

        self.embedding = nn.Embedding(self.H_o, self.H_i)

        self.lstm = nn.LSTM(
            input_size=self.H_i,
            hidden_size=self.H_c,
            num_layers=self.N_LAYERS,
            batch_first=True,
            dropout=dropout * int((self.N_LAYERS > 1)),
            bidirectional=(self.D == 2),
            proj_size=0
        )
        self.full = nn.Linear(self.H_c * self.D, self.H_o)

    def forward(self, x, h0=None, c0=None):
        if h0 == None or c0 == None:
            h0 = torch.zeros(self.D * self.N_LAYERS, x.shape[0], self.H_c)
            c0 = torch.zeros(self.D * self.N_LAYERS, x.shape[0], self.H_c)

        x = self.embedding(x)
        self.lstm(x, (h0, c0))
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.full(out[:, -1, :])/self.T
        return out, hn, cn # logits, hidden, cell