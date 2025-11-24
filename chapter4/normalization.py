import torch
from layer_norm import LayerNorm

torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = torch.nn.Sequential(torch.nn.Linear(5, 6), torch.nn.ReLU())
out = layer(batch_example)
print(out)

mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
out_norm = (out - mean) / torch.sqrt(var)
print("Normalized output:", out_norm)

mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)

print("Mean after normalization (should be close to 0):", mean)
print("Variance after normalization (should be close to 1):", var)

ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)

print("Mean after LayerNorm (should be close to 0):", mean)
print("Variance after LayerNorm (should be close to 1):", var)
