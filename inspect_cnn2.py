
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from model import GPT, GPTConfig
from transformers import AutoTokenizer


# ============================================================
# Konfiguracja
# ============================================================

checkpoint_path = "outs/cnn-transformer-5000-k3.pt"
tokenizer_name = "speakleash/Bielik-1.5B-v3"
val_path = Path("data/speakleash/val.bin")
output_dir = Path("cnn_analysis")
output_dir.mkdir(exist_ok=True)

num_documents = 500
num_filters = 10
random_seed = 42
num_filters_to_inspect = 100
top_k = 5


# ============================================================
# Ładowanie modelu
# ============================================================

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

model_args = checkpoint["model_args"].copy()

if "use_cnn" in model_args:
    model_args["use_cnn_input"] = model_args.pop("use_cnn")
    model_args["use_cnn_block"] = False
    model_args["use_cnn_residual"] = False

model = GPT(GPTConfig(**model_args))

state_dict = {
    key.removeprefix("_orig_mod."): value
    for key, value in checkpoint["model"].items()
}

model.load_state_dict(state_dict)
model.eval()

if model.cnn is None:
    raise ValueError("Checkpoint nie zawiera CNN.")

conv = model.cnn.conv
n_filters = conv.out_channels

print(f"Parametry modelu: {model.get_num_params() / 1e6:.2f}M")
print(f"Liczba filtrów CNN: {n_filters}")


# ============================================================
# Wczytanie dokumentów
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

val_ids = np.fromfile(val_path, dtype=np.uint16)
eos_id = tokenizer.eos_token_id

documents = []
current_doc = []

for token_id in val_ids:
    current_doc.append(int(token_id))

    if token_id == eos_id:
        if len(current_doc) > 150:
            documents.append(current_doc)
        current_doc = []

if len(current_doc) > 150:
    documents.append(current_doc)

random.seed(random_seed)

selected_docs = random.sample(
    documents,
    min(num_documents, len(documents))
)

print(f"Wybrano dokumentów: {len(selected_docs)}")


# ============================================================
# Przechwycenie aktywacji CNN
# ============================================================

all_activations = []
all_tokens = []
doc_ranges = []

activations = {}

def save_activation(module, inputs, output):
    activations["cnn"] = output.detach().cpu()


hook = conv.register_forward_hook(save_activation)

try:
    with torch.no_grad():
        for doc_id, doc in enumerate(selected_docs):

            doc = doc[:model.config.block_size]

            tokens = torch.tensor([doc], dtype=torch.long)

            model(tokens)

            # (liczba filtrów, długość sekwencji)
            current_activation = activations["cnn"][0]

            start = len(all_tokens)

            all_activations.append(current_activation)
            all_tokens.extend(doc)

            doc_ranges.append({
                "start": start,
                "end": len(all_tokens),
                "doc_id": doc_id,
            })

finally:
    hook.remove()


# Łączenie aktywacji wszystkich dokumentów.
activation = torch.cat(all_activations, dim=1).numpy()

token_strings = tokenizer.convert_ids_to_tokens(all_tokens)

print(f"Liczba analizowanych tokenów: {len(all_tokens)}")


# ============================================================
# Wybór najbardziej aktywnych filtrów
# ============================================================

filter_stats = []

for filter_id in range(n_filters):

    values = activation[filter_id]

    filter_stats.append({
        "filter_id": filter_id,
        "mean": np.mean(values),
        "std": np.std(values),
        "mean_abs": np.mean(np.abs(values)),
        "max": np.max(values),
        "p95": np.percentile(values, 95),
        "positive_ratio": np.mean(values > 0),
    })

stats_df = pd.DataFrame(filter_stats)

# Wybieramy filtry o największej średniej bezwzględnej aktywacji.
selected_filters = (
    stats_df.nlargest(num_filters, "mean_abs")["filter_id"].tolist()
)

print("\nWybrane filtry:")

print(
    stats_df[stats_df["filter_id"].isin(selected_filters)]
    .sort_values("mean_abs", ascending=False)
    .to_string(index=False)
)

stats_df.to_csv(
    output_dir / "filter_statistics.csv",
    index=False
)

# Aktywacje tylko wybranych filtrów.
selected = activation[selected_filters]


# ============================================================
# Pomocnicza funkcja: znajdowanie dokumentu i kontekstu
# ============================================================

def get_context(position, window=3):
    doc_range = next(
        r for r in doc_ranges
        if r["start"] <= position < r["end"]
    )

    start = max(doc_range["start"], position - window)
    end = min(doc_range["end"], position + window + 1)

    context_tokens = token_strings[start:end]

    context = tokenizer.convert_tokens_to_string(context_tokens)

    target_token = token_strings[position]

    return context, target_token


# ============================================================
# 1. Mapa cieplna aktywacji
# ============================================================

limit = float(np.percentile(np.abs(selected), 99))
limit = max(limit, 1e-8)

fig, ax = plt.subplots(figsize=(16, 6))

image = ax.imshow(
    selected,
    aspect="auto",
    interpolation="nearest",
    cmap="coolwarm",
    vmin=-limit,
    vmax=limit,
)

ax.set_yticks(range(len(selected_filters)))
ax.set_yticklabels(selected_filters)

ax.set_ylabel("Indeks filtra")
ax.set_xlabel("Pozycja tokenu")
ax.set_title("Aktywacje najbardziej aktywnych filtrów CNN")

fig.colorbar(image, ax=ax, label="Aktywacja")

fig.tight_layout()
fig.savefig(output_dir / "activation_heatmap.png", dpi=180)
plt.close(fig)


# ============================================================
# 2. Aktywacje poszczególnych filtrów
# ============================================================

fig, axes = plt.subplots(
    len(selected_filters),
    1,
    figsize=(14, 2 * len(selected_filters)),
    sharex=True,
)

if len(selected_filters) == 1:
    axes = [axes]

for ax, filter_id, values in zip(
    axes, selected_filters, selected
):
    ax.plot(values, linewidth=0.8)
    ax.axhline(0, linewidth=0.6)
    ax.set_ylabel(f"Filtr {filter_id}")
    ax.grid(True, alpha=0.2)

axes[-1].set_xlabel("Pozycja tokenu")

fig.tight_layout()
fig.savefig(output_dir / "activation_lines.png", dpi=180)
plt.close(fig)


# ============================================================
# 3. Najsilniejsze aktywacje pierwszych 100 filtrów
# ============================================================

num_filters_to_inspect = min(100, n_filters)

top_rows = []

report_path = output_dir / "first_100_filters.txt"

with open(report_path, "w", encoding="utf-8") as report:

    for filter_id in range(num_filters_to_inspect):

        values = activation[filter_id]

        # 5 najsilniejszych aktywacji dodatnich
        positive_positions = np.argsort(values)[-top_k:][::-1]

        # 5 najsilniejszych aktywacji ujemnych
        negative_positions = np.argsort(values)[:top_k]

        report.write(f"\n{'=' * 60}\n")
        report.write(f"FILTR {filter_id}\n")
        report.write(f"{'=' * 60}\n")

        for kind, positions in [
            ("positive", positive_positions),
            ("negative", negative_positions),
        ]:
            report.write(f"\n{kind.upper()} ACTIVATIONS\n")

            for pos in positions:
                pos = int(pos)

                row = {
                    "filter_id": filter_id,
                    "type": kind,
                    "position": pos,
                    "token": token_strings[pos],
                    "activation": float(values[pos]),
                    "context": get_context(pos),
                }

                top_rows.append(row)

                line = (
                    f"{kind:>8} | "
                    f"activation={row['activation']:.4f} | "
                    f"token={row['token']!r} | "
                    f"context={row['context']!r}"
                )

                report.write(line + "\n")

top_df = pd.DataFrame(top_rows)

top_df.to_csv(
    output_dir / "first_100_filters.csv",
    index=False,
    encoding="utf-8",
)

print(f"\nPrzeanalizowano filtrów: {num_filters_to_inspect}")
print(f"Liczba przykładów: {len(top_df)}")
print("Zapisano:")
print(f" - {report_path}")
print(f" - {output_dir / 'first_100_filters.csv'}")