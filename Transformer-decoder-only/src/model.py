import math
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F

# Huperparameters
class ModelConfig:
    """Configuration for the model"""
    CONTEXT_LENGTH: int = 16   # Number of context frames
    D_MODEL: int = 64          # Dimension of the model
    D_FF: int = 256            # Dimension of the feed forward network
    NUM_BLOCKS: int = 8        # Number of transformer blocks
    NUM_HEADS: int = 4         # Number of heads in multi-head attention
    DROP_OUT: float = 0.1      # Drop out rate
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

config = ModelConfig()


class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(config.D_MODEL, config.D_FF),
            nn.ReLU(),
            nn.Linear(config.D_FF, config.D_MODEL),
            nn.Dropout(config.DROP_OUT)
        )
    
    def forward(self, x):
        return self.ffn(x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(config.D_MODEL, config.D_MODEL // config.NUM_HEADS, bias=False)
        self.wk = nn.Linear(config.D_MODEL, config.D_MODEL // config.NUM_HEADS, bias=False)
        self.wv = nn.Linear(config.D_MODEL, config.D_MODEL // config.NUM_HEADS, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(config.CONTEXT_LENGTH, config.CONTEXT_LENGTH)))
        self.drop_out = nn.Dropout(config.DROP_OUT)
        
    def forward(self, x, mask=None):
        B, T, C = x.shape # B: batch size, T: number of tokens, C: dimension of the model
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        # Compute the scaled dot product attention
        weights = (q @ k.transpose(-2, -1)) / math.sqrt(config.D_MODEL // config.NUM_HEADS)
        weights = weights.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.drop_out(weights)
        
        output = weights @ v

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([ScaledDotProductAttention() for _ in range(config.NUM_HEADS)])
        self.projection_layer = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.drop_out = nn.Dropout(config.DROP_OUT)

    def forward(self, x):
        # Apply the attention heads
        attention_heads = [head(x) for head in self.heads]
        attention_heads = torch.cat(attention_heads, dim=-1)
        
        # Apply the projection layer
        output = self.projection_layer(attention_heads)
        output = self.drop_out(output)
        
        return output

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.D_MODEL)
        self.layer_norm2 = nn.LayerNorm(config.D_MODEL)
        self.multi_head_attention = MultiHeadAttention()
        self.feed_forward_network = FeedForwardNetwork()

    def forward(self, x):
        x = x + self.multi_head_attention(self.layer_norm1(x))
        x = x + self.feed_forward_network(self.layer_norm2(x))
        return x

class Model(nn.Module):
    def __init__(self,  max_token_value=100256): # default tiktoken cl100k vocab size
        super(Model, self).__init__()
        self.embedding = nn.Embedding(max_token_value, config.D_MODEL)
        self.transformer_blocks = nn.Sequential(*(
            [TransformerBlock() for _ in range(config.NUM_BLOCKS)] +
            [nn.LayerNorm(config.D_MODEL)]
        ))
        self.model_out_linear_layer = nn.Linear(config.D_MODEL, max_token_value)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Embed the input tokens and add positional encoding
        positional_encoding_matrix = torch.zeros((config.CONTEXT_LENGTH, config.D_MODEL)).to(config.DEVICE)
        position = torch.arange(0, config.CONTEXT_LENGTH, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.D_MODEL, 2).float() * (-math.log(10000.0) / config.D_MODEL))
        positional_encoding_matrix[:, 0::2] = torch.sin(position * div_term)
        positional_encoding_matrix[:, 1::2] = torch.cos(position * div_term)
        positional_embedding = positional_encoding_matrix[:T, :].to(config.DEVICE)
        x = self.embedding(idx) + positional_embedding

        # Pass the input through the transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # get the final logits
        logits = self.model_out_linear_layer(x)

        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens=100):
        # idx is (B, T) array of token in the current context
        for _ in range(max_new_tokens):
            # crop the context to the last CONTEXT_LENGTH tokens
            idx_crop = idx[:, -min(config.CONTEXT_LENGTH, idx.size(1)):]
            # get the prediction for the next token
            logits, _ = self.forward(idx_crop)
            # get the last time step from the logits where the dimension of the logits are (B, T, C)
            logits_last_timestep = logits[:, -1, :]
            # apply softmax to the logits
            probs = F.softmax(logits_last_timestep, dim=-1)
            # get the token with the highest probability
            idx_new_token = torch.multinomial(probs, num_samples=1)
            # append the new token to the context
            idx = torch.cat((idx, idx_new_token), dim=1)
        
        return idx