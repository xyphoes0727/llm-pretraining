import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import N_VOCAB, MAX_SEQ_LEN
import math
from rope import apply_rotary_emb, precompute_freqs_cis

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, value_dim: int, qk_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.value_dim = value_dim
        self.qk_dim = qk_dim
        self.embed_dim = embed_dim
        self.norm_param = math.sqrt(qk_dim)
        self.dropout = nn.Dropout(p=0.1)

        self.W_Q = nn.Linear(embed_dim, n_heads*qk_dim)
        nn.init.xavier_normal_(self.W_Q.weight)

        self.W_K = nn.Linear(embed_dim, n_heads*qk_dim)
        nn.init.xavier_normal_(self.W_K.weight)

        self.W_V = nn.Linear(embed_dim, n_heads*value_dim)
        nn.init.xavier_normal_(self.W_V.weight)

        self.W_O = nn.Linear(n_heads*value_dim, embed_dim)
        nn.init.xavier_normal_(self.W_O.weight)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, padding_mask: torch.Tensor):  # x: (B,n,d_emb)
        Q = self.W_Q(x)  # (B, n, h*d_qk)
        K = self.W_K(x)  # (B, n, h*d_qk)
        V: torch.Tensor = self.W_V(x)  # (B, n, h*d_value)
        # print(f"Q: {Q.shape}")

        # Apply RoPE per-head on qk_dim
        Q = Q.reshape(Q.shape[0], Q.shape[1],
                      self.n_heads, self.qk_dim)  # (B,n,h,d)
        K = K.reshape(K.shape[0], K.shape[1], self.n_heads, self.qk_dim)
        V = V.reshape(V.shape[0], V.shape[1], self.n_heads, self.value_dim)
        Q, K = apply_rotary_emb(Q, K, freqs_cis)

        # (B,h,n,d)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # (B,h,n,n) !!! mT reverses only last 2 dims
        softmax_scores = Q @ (K.mT) / self.norm_param
        n = x.shape[1]
        # print(f"softmax scores: {softmax_scores.shape}")
        causal_mask = torch.triu(torch.ones(n, n, device=device), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        softmax_scores = softmax_scores.masked_fill_(
            causal_mask == 1, float('-inf'))

        # padding_mask is (B,n)
        padding_mask_key = padding_mask.unsqueeze(1).unsqueeze(1)  # (B,1,1,n)
        # VV IMP- We need key and query padding masks as only masking the key pads will mean-
        # we have masked the columns by key padding, but the rows contain query tokens that
        # are pad tokens. To mask their attentions, we have to use a mask that operates on the
        # query dimension also. So, we need both key and query padding masks.

        softmax_scores = softmax_scores.masked_fill_(
            padding_mask_key, float('-inf'))

        attention_weights = torch.softmax(softmax_scores, dim=-1)  # (B,h,n,n)
        attention_weights = self.dropout(attention_weights)
        attention_scores = attention_weights @ V  # (B,h,n,d_v)

        # print(f"softmax_scores: {softmax_scores.shape}, attention_weights: {attention_weights.shape}, attention_scores: {attention_scores.shape}\n")
        attention_scores = attention_scores.permute(0, 2, 1, 3)
        # print(f"after permute:{attention_scores.shape}")
        attention_scores = attention_scores.flatten(2, 3)  # (B,n,h*d_v)
        # print(f"after flatten:{attention_scores.shape}")

        final_scores = self.W_O(attention_scores)  # (B,n,d_e)

        # !!! We apply padding mask query to final attention since applying ot along on
        # the softmax itself will lead to 0/0 = inf attention score
        padding_mask_query = padding_mask.unsqueeze(-1)   # (B,n,1)
        final_scores = final_scores.masked_fill_(padding_mask_query, 0.)

        # print(f"final_scores: {final_scores.shape}")
        return final_scores  # (B,n, d_emb)


class FFN(nn.Module):  # input is (n, d_v)
    def __init__(self, input_dim: int, h1_dim: int = 3072):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, h1_dim)
        self.dropout = nn.Dropout(p=0.1)m2
        self.h1 = nn.Linear(h1_dim, input_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.h1(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_emb: int, n_heads: int, d_val: int, d_qk: int):
        super().__init__()
        self.d_emb = d_emb
        self.n_heads = n_heads
        self.d_val = d_val
        self.d_qk = d_qk

        self.ln1 = nn.LayerNorm(d_emb)
        self.drop1 = nn.Dropout(p=0.1)
        self.mha = MultiHeadAttention(d_emb, n_heads, d_val, d_qk)
        self.drop2 = nn.Dropout(p=0.1)
        self.ln2 = nn.LayerNorm(d_emb)
        self.ffn = FFN(d_emb)

    def forward(self, x, freqs_cis, padding_mask):
        # (B,n,d_e) throughout the layers
        x = self.ln1(x)  #
        x_ = x
        x = self.mha(x, freqs_cis, padding_mask)

        # dropout is applied on entire tensor after residual connection
        x = self.drop1(x+x_)

        x = self.ln2(x)
        x_ = x

        x = self.ffn(x)
        x = x.masked_fill_(padding_mask.unsqueeze(-1), 0.)
        x = self.drop2(x+x_)
        return x


class MyModel(nn.Module):
    def __init__(self, d_emb: int, n_heads: int, d_val: int, d_qk: int, n_blocks: int = 12):
        super().__init__()
        self.d_emb = d_emb
        self.tok_emb = nn.Embedding(
            N_VOCAB, d_emb, device=device, padding_idx=0)
        self.register_buffer(
            "freq_cis", precompute_freqs_cis(d_qk, MAX_SEQ_LEN))
        self.blocks = nn.ModuleList([TransformerBlock(
            d_emb, n_heads, d_val, d_qk
        ) for _ in range(n_blocks)])

        self.ln_f = nn.LayerNorm(d_emb)
        self.lin = nn.Linear(d_emb, N_VOCAB)
        self.lin.weight = self.tok_emb.weight

    def forward(self, x):
        padding_mask = (x <= 0)
        # print(f"x: {x.shape}")

        x = self.tok_emb(x)  # (B,n) -> (B,n,d_e)
        # print(f"x_embedded: {x.shape}")
        seq_len = x.shape[1]
        freqs_cis = self.freq_cis[:seq_len].to(device=x.device)
        # print(f"freq_cis: {freqs_cis.shape}")

        for block in self.blocks:
            x = block(x, freqs_cis, padding_mask=padding_mask)

        x = self.ln_f(x)
        x = self.lin(x)  # (B,n,N_VOCAB)
        return x / math.sqrt(self.d_emb)

