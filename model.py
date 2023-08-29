import torch
import torch.nn as nn
import torch.nn.functional as F
import json

#hyperparameters
config = json.load(open("./config.json"))
layers = config["layers"]
heads = config["heads"]
d_model = config["d_model"]
batch_size = config["batch_size"]
vocab_size = config["vocab_size"]
block_length = config["block_length"]
activation = config["activation"]
attention = config["attention"]
groups = config["groups"]
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

#model
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(d_model, 3*d_model, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_length, block_length)).view(1,1,block_length,block_length))

    def forward(self, x): # (b, t, c)
        b, t, c = x.size()
        q, k, v = self.proj(x).split(d_model, dim=-1)
        q = q.view(b, t, heads, int(c/heads)).transpose(1,2)
        k = k.view(b, t, heads, int(c/heads)).transpose(1,2) # (b, h, t, c/h)
        v = v.view(b, t, heads, int(c/heads)).transpose(1,2)

        weights = (q @ k.transpose(-2,-1)) * (k.size(dim=-1)**-0.5) # (b, h, t, t)
        weights = weights.masked_fill(self.mask[:,:,:t,:t] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        out = weights @ v # (b, h, t, c/h)
        out = out.transpose(1, 2).contiguous().view(b,t,c) # (b, t, c)
        return out
    
class GroupedQueryAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_q = nn.Linear(d_model, d_model, bias=False)
        self.proj_k = nn.Linear(d_model, int(d_model * groups / heads), bias=False)
        self.proj_v = nn.Linear(d_model, int(d_model * groups / heads), bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_length, block_length)).view(1,1,block_length,block_length))


    def forward(self, x):
        b, t, c = x.size()
        q = self.proj_q(x).view(b, t, heads, int(c/heads)).transpose(1,2) #(b, h, t, c/h)
        k = self.proj_k(x).view(b, t, groups, int(c/heads)).transpose(1,2) #(b, g, t, c/h)
        v = self.proj_v(x).view(b, t, groups, int(c/heads)).transpose(1,2) #(b, g, t, c/h)

        h2g = heads // groups
        weights = torch.cat([q[:,h2g * i:h2g * (i+1),:,:] @ k[:,i,:,:].transpose(-2,-1).view(b,1,c//heads,t) for i in range(k.size(dim=1))], dim=1)
        weights = weights * (k.size(dim=-1)**-0.5) #(b, h, t, t)
        weights = weights.masked_fill(self.mask[:,:,:t,:t] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        out = torch.cat([weights[:,h2g *i:h2g*(i+1),:,:] @ v[:,i,:,:].view(b,1,t,c//heads) for i in range(v.size(dim=1))], dim=1) # (b, h, t, c/h)
        out = out.transpose(1, 2).contiguous().view(b,t,c) #(b, t, c)

        return out


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffw1 = nn.Linear(d_model, 4*d_model, bias=False)
        self.ffw2 = nn.Linear(4*d_model, d_model, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x): # (b, t, c)
        out = self.ffw1(x)
        out = self.relu(out)
        out = self.ffw2(out)
        return out
    
class FeedForwardSwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(d_model, int(4*d_model*(2/3)), bias=False) #scaling the hidden units to keep computation about constant
        self.V = nn.Linear(d_model, int(4*d_model*(2/3)), bias=False)
        self.W2 = nn.Linear(int(4*d_model*2/3), d_model, bias=False)
    
    def forward(self, x):
        out = F.silu(self.W1(x)) * self.V(x)
        out = self.W2(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention() if config["attention"] == "MHA" else GroupedQueryAttention()
        self.ffw = FeedForward() if config["activation"] == "ReLU" else FeedForwardSwiGLU()
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
        print(f"Using {config['attention']} and {config['activation']}", '\n')
        self.tok_emb = nn.Embedding(vocab_size, d_model) 
        self.pos_emb = nn.Embedding(block_length, d_model)
        self.blocks = nn.ModuleList(TransformerBlock() for _ in range(layers))
        self.ffw = nn.Linear(d_model, vocab_size, bias=False)


    def forward(self, x, y=None):
        pos = torch.arange(x.size(-1)).to(device)
        x = self.tok_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        out = self.ffw(x)

        if y is not None:
            logits = out.view(batch_size*block_length, vocab_size)
            targets = y.view(batch_size*block_length)
            loss = F.cross_entropy(logits, targets)
            return out, loss
        else:
            return out
        
    def generate(self, x, length):
        x = x.view(1,-1)
        for _ in range(length):
            x = x[:, -block_length:]
            out = self(x)
            out = out[:, -1, :]
            probs = F.softmax(out, dim=-1)
            x1 = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, x1), dim=1)
        return x