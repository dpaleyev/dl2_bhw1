import torch.nn as nn
import torch
import math
import copy

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return token_embedding + self.pos_embedding[:token_embedding.size(0), :]

class PreNormDecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, hidden_dim):
        super().__init__()

        self.attention = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        self.attn_layernorm = nn.LayerNorm(embed_dim)
        self.ffn_layernorm = nn.LayerNorm(embed_dim)
    
    def forward(self, x, attention_mask, padding_mask):
        x_norm = self.attn_layernorm(x)
        x_attn, _ = self.attention(x_norm, x_norm, x_norm, key_padding_mask=padding_mask, attn_mask=attention_mask)
        h = x_attn + x
        out = h + self.ffn(self.ffn_layernorm(x))
        return out

class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decoder = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
    
    def forward(self, x, attention_mask, padding_mask):
        for layer in self.decoder:
            x = layer(x, attention_mask, padding_mask)
        return x

class LLaMA(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, hidden_dim, num_layers, max_seq_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)
        self.decoder = Decoder(PreNormDecoderBlock(embed_dim, n_heads, hidden_dim), num_layers)
        self.linear = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x, attention_mask, padding_mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.decoder(x, attention_mask, padding_mask)
        x = self.linear(x)
        return x
