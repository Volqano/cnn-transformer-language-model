
"""
Compare token embedding nearest neighbors in Transformer-only
and CNN+Transformer nanoGPT checkpoints.
"""

import argparse
import os

import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer

from model import GPTConfig, GPT


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--transformer",
    default="outs/transformer-5000.pt",
    help="Transformer-only checkpoint",
)
parser.add_argument(
    "--cnn",
    default="outs/cnn-transformer-k3-5000.pt",
    help="CNN+Transformer checkpoint",
)
parser.add_argument("--top_k", type=int, default=10)
parser.add_argument("--output", default="outs/embedding_comparison.csv")
args = parser.parse_args()

device = "cpu"

# Words to investigate. The leading space helps tokenize words
# as they often appear inside a sentence.
words = [
    " Warszawa",
    " Piłka",
    " Wrocław",
    " Polska",
    " król",
    " królestwo",
    " pies",
    " kot",
    " komputer",
    " zamek",
]


# ------------------------------------------------------------
# Load tokenizer
# ------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    "speakleash/Bielik-1.5B-v3",
    use_fast=True,
)

print("Tokenizer vocabulary size:", len(tokenizer))


# ------------------------------------------------------------
# Load model checkpoint
# ------------------------------------------------------------

def load_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    config = GPTConfig(**checkpoint["model_args"])
    model = GPT(config)

    state_dict = checkpoint["model"]

    # Handle checkpoints saved using torch.compile()
    prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(prefix):
            state_dict[key[len(prefix):]] = state_dict.pop(key)

    model.load_state_dict(state_dict)
    model.eval()

    print(
        f"Loaded {checkpoint_path}: "
        f"vocab_size={config.vocab_size}, "
        f"n_embd={config.n_embd}"
    )

    if config.vocab_size != len(tokenizer):
        raise ValueError(
            "Tokenizer vocabulary size does not match model vocabulary!"
        )

    return model


transformer = load_model(args.transformer)
cnn_transformer = load_model(args.cnn)


# ------------------------------------------------------------
# Extract and normalize token embeddings
# ------------------------------------------------------------

def get_normalized_embeddings(model):
    # Shape: (vocab_size, embedding_dim)
    embeddings = model.transformer.wte.weight.detach().float()

    # Normalize each row so dot product equals cosine similarity
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu()


emb_transformer = get_normalized_embeddings(transformer)
emb_cnn = get_normalized_embeddings(cnn_transformer)

if emb_transformer.shape != emb_cnn.shape:
    raise ValueError("The two models have different embedding shapes!")


# ------------------------------------------------------------
# Find nearest tokens
# ------------------------------------------------------------

def display_token(token_id):
    text = tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
    )
    return repr(text)


def nearest_tokens(embeddings, token_id, top_k=10):
    query = embeddings[token_id]

    # Cosine similarities, because all vectors are normalized
    similarities = embeddings @ query

    # Do not return the queried token itself
    similarities[token_id] = -float("inf")

    scores, indices = torch.topk(
        similarities,
        k=min(top_k, embeddings.shape[0] - 1),
    )

    return [
        f"{display_token(idx)} ({score:.3f})"
        for idx, score in zip(indices.tolist(), scores.tolist())
    ]


# ------------------------------------------------------------
# Compare the two models
# ------------------------------------------------------------

rows = []

for word in words:
    token_ids = tokenizer.encode(
        word,
        add_special_tokens=False,
    )

    print(f"\n{'=' * 90}")
    print(f"WORD: {word!r}")
    print("Tokenization:", [
        (token_id, display_token(token_id))
        for token_id in token_ids
    ])

    for token_id in token_ids:
        token_piece = display_token(token_id)

        transformer_neighbors = nearest_tokens(
            emb_transformer,
            token_id,
            args.top_k,
        )

        cnn_neighbors = nearest_tokens(
            emb_cnn,
            token_id,
            args.top_k,
        )

        print(f"\nToken {token_id}: {token_piece}")
        print("  Transformer-only:")
        for neighbor in transformer_neighbors:
            print("   ", neighbor)

        print("  CNN + Transformer:")
        for neighbor in cnn_neighbors:
            print("   ", neighbor)

        rows.append({
            "word": word,
            "token_id": token_id,
            "token_piece": token_piece,
            "transformer_neighbors": " | ".join(
                transformer_neighbors
            ),
            "cnn_transformer_neighbors": " | ".join(
                cnn_neighbors
            ),
        })


df = pd.DataFrame(rows)

os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
df.to_csv(args.output, index=False, encoding="utf-8-sig")

print(f"\nResults saved to: {args.output}")
