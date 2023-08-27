import torch
import torch.nn as nn
import torch.nn.functional as F

#hyperparameters
layers = 6
heads = 8
d_model = 512
batch_size = 32
tokens = 2000
block_length = 40
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
#model

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(d_model, 3*d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_length, block_length)).view(1,1,block_length,block_length))

    def forward(self, x): # (b, t, c)
        b, t, c = x.size()
        q, k, v = self.proj(x).split(d_model, dim=-1)
        q = q.view(b, t, heads, int(c/heads)).transpose(1,2)
        k = k.view(b, t, heads, int(c/heads)).transpose(1,2) # (b, h, t, c/h)
        v = v.view(b, t, heads, int(c/heads)).transpose(1,2)

        dk = k.size(dim=-1)
        weights = (q @ k.transpose(-2,-1)) * (dk**-0.5) # (b, h, t, t)
        weights = weights.masked_fill(self.mask[:,:,:t,:t] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        out = weights @ v # (b, h, t, c/h)
        out = out.transpose(1, 2).contiguous().view(b,t,c) # (b, t, c)
        #out = self.w_out(out)
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
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        out = self.attn(self.ln1(x)) + x
        out = self.ffw(self.ln2(out)) + out
        #out = self.attn(x)
        return out

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(tokens, d_model) 
        self.pos_emb = nn.Embedding(block_length, d_model)
        self.blocks = nn.ModuleList(TransformerBlock() for _ in range(layers))
        self.ffw = nn.Linear(d_model, tokens)


    def forward(self, x, y=None):
        pos = torch.arange(x.size(-1)).to(device)
        x = self.tok_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        out = self.ffw(x)

        if y is not None:
            logits = out.view(batch_size*block_length, tokens)
            targets = y.view(batch_size*block_length)
            loss = F.cross_entropy(logits, targets)
            return out, loss
        else:
            return out
        
    def generate(self, x):
        x = x.view(1,1)
        for _ in range(150):
            x = x[:, -block_length:]
            out = self(x)
            out = out[:, -1, :]
            probs = F.softmax(out, dim=-1)
            x1 = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, x1), dim=1)
        return x