import torch
import torch.nn as nn
import torch.nn.functional as F

#hyperparameters
layers = 8
heads = 8
d_model = 128
batch_size = 128

#model

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)

    def forward(self, x): # (b, t, c)
        b, t, c = x.size()
        q = self.q_proj(x).view(b, t, heads, int(c/heads)).transpose(1,2)
        k = self.k_proj(x).view(b, t, heads, int(c/heads)).transpose(1,2) # (b, h, t, c/h)
        v = self.v_proj(x).view(b, t, heads, int(c/heads)).transpose(1,2)

        dk = k.size(dim=3)
        weights = F.softmax(q @ (k.transpose(2,3)) / (dk ** (1/2)), dim=-1) # (b, h, t, t)
        out = weights @ v # (b, h, t, c/h)
        out = out.transpose(1, 2).contiguous().view(b,t,c) # (b, t, c)
        out = self.w_out(out)
        return out
        
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffw1 = nn.Linear(d_model, 4*d_model)
        self.ffw2 = nn.Linear(4*d_model, d_model)
        self.relu = nn.ReLU()

    def forward(self, x): # (b, t, c)
        out = self.ffw1(x)
        out = self.relu(out)
        out = self.ffw2(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.ffw = FeedForward()
        ln1 = nn.LayerNorm()
        ln2 = nn.LayerNorm()

    def forward(self, x):
        sl1 = self.ln1(self.attn(x) + x) # (sl1 = sub-layer 1)
        sl2 = self.ln2(self.ffw(sl1) + sl1)
        return sl2
    

MHA = FeedForward()
x = torch.randn(10, 20, 128)
out = MHA(x)
print(out.shape)