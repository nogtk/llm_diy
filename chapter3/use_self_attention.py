import torch

from casual_attention import CasualAttention
from multi_ahead_attention import MultiHeadAttentionWrapper
from self_attention_v1 import SelfAttentionV1
from self_attention_v2 import SelfAttentionV2

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55],
    ]
)

d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)
sa_v1 = SelfAttentionV1(d_in, d_out)
print(sa_v1(inputs))

torch.manual_seed(789)
sa_v2 = SelfAttentionV2(d_in, d_out)
print(sa_v2(inputs))

torch.manual_seed(123)
batch = torch.stack([inputs, inputs], dim=0)
context_length = batch.shape[1]
ca = CasualAttention(d_in, d_out, context_length, dropout=0.0)
context_vecs = ca(batch)
print("context_vecs shape:", context_vecs)

print("================================")
torch.manual_seed(123)
context_length = batch.shape[1]
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout=0.0, num_heads=2)
context_vecs = mha(batch)

print(context_vecs)
print("context_vecs shape:", context_vecs.shape)
