import os
import sys

import tiktoken
import torch

# 現在のディレクトリをパスに追加
sys.path.append(os.path.dirname(__file__))
from util import text_to_token_ids, token_ids_to_text

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from chapter2.gpt_dataset_v1 import create_dataloader_v1
from chapter4.generate_text_simple import generate_text_simple
from chapter4.gpt_model import GPTModel
from chapter5.adjust_model import load_weights_into_gpt
from chapter5.get_download import download_and_load_gpt2
from chapter5.new_text_generator import generate

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"],
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters in text:", total_characters)
print("Tokens in text:", total_tokens)

train_ratio = 0.90
split_idx = int(train_ratio * total_characters)
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0,
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=False,
    num_workers=0,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
# model = GPTModel(GPT_CONFIG_124M)
# model.load_state_dict(checkpoint["model_state_dict"])
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
# optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# model.train()

num_epochs = 10
# train_losses, val_losses, tokens_seen = train_model_simple(
#     model,
#     train_loader,
#     val_loader,
#     optimizer,
#     device,
#     num_epochs=num_epochs,
#     eval_freq=5,
#     eval_iter=5,
#     start_context="Every effort moves you",
#     tokenizer=tokenizer,
# )

# torch.save(
#     {
#         "model_state_dict": model.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#     },
#     "model_and_optimizer.pth",
# )


# model.to("cpu")
# model.eval()

# tokenizer = tiktoken.get_encoding("gpt2")
# token_ids = generate_text_simple(
#     model=model,
#     idx=text_to_token_ids("Every effort moves you", tokenizer),
#     max_new_tokens=25,
#     context_size=GPT_CONFIG_124M["context_length"],
# )
# print("Output text after training(simple):\n", token_ids_to_text(token_ids, tokenizer))

# torch.manual_seed(123)
# token_ids_new = generate(
#     model=model,
#     idx=text_to_token_ids("Every effort moves you", tokenizer),
#     max_new_tokens=15,
#     context_size=GPT_CONFIG_124M["context_length"],
#     temperature=1.4,
#     top_k=25,
# )
# print("Output text after training(new):\n", token_ids_to_text(token_ids_new, tokenizer))

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(
    {
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12,
        "context_length": 1024,
        "qkv_bias": True,
    }
)
gpt = GPTModel(NEW_CONFIG)
gpt.eval()

load_weights_into_gpt(gpt, params)
gpt.to(device)

torch.manual_seed(123)
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    temperature=1.5,
    top_k=50,
)
print("Output text with adjusted model:\n", token_ids_to_text(token_ids, tokenizer))
