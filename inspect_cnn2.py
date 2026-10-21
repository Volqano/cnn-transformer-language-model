
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from model import GPT, GPTConfig
from transformers import AutoTokenizer


# ============================================================
# Konfiguracja
# ============================================================

checkpoint_path = "outs/cnn-transformer-5000-k3.pt"
tokenizer_name = "speakleash/Bielik-1.5B-v3"
output_dir = Path("cnn_analysis")
output_dir.mkdir(exist_ok=True)

# 10 filtrów do analizy — możesz zmienić ich indeksy.
filter_ids = [132, 292, 340, 125, 129, 185, 307, 246, 188, 100]

# Maksymalna liczba tokenów analizowanego tekstu.
max_tokens = 800

# Liczba najsilniejszych aktywacji dodatnich i ujemnych,
# które wypiszemy dla każdego filtra.
top_k = 5


# Dłuższy tekst testowy. Możesz zastąpić go własnym fragmentem korpusu.
text = """
Język polski należy do grupy języków zachodniosłowiańskich i jest używany
przede wszystkim w Polsce. Jego współczesna forma rozwijała się przez wiele
stuleci pod wpływem zmian społecznych, politycznych i kulturowych. Słownictwo
polszczyzny zawiera wyrazy o różnym pochodzeniu, a gramatyka wykorzystuje
między innymi przypadki, rodzaje gramatyczne i rozbudowaną odmianę czasowników.

W informatyce modele językowe uczą się przewidywać kolejne tokeny na podstawie
kontekstu. Podczas treningu model otrzymuje sekwencje tekstu i dostosowuje swoje
parametry tak, aby zmniejszać błąd predykcji. Reprezentacje wektorowe pozwalają
modelowi kodować informacje o tokenach, natomiast mechanizm uwagi umożliwia
łączenie informacji z różnych pozycji sekwencji.

Konwolucja jednowymiarowa przetwarza lokalne fragmenty sekwencji. Każdy filtr
łączy informacje z niewielkiego okna wejściowego za pomocą wyuczonych wag.
W konwolucji przyczynowej wynik dla danej pozycji zależy tylko od bieżącego
tokenu i wcześniejszego kontekstu, co jest istotne przy autoregresyjnym
przewidywaniu tekstu. Rozmiar jądra określa, ile sąsiednich pozycji wpływa
bezpośrednio na wynik filtra.

W eksperymencie naukowym warto sprawdzić, czy poszczególne filtry wykazują
powtarzalne wzorce aktywacji. Można porównywać ich odpowiedzi na różnych
fragmentach tekstu, analizować lokalny kontekst najsilniejszych reakcji oraz
mierzyć średnie wartości aktywacji na różnych pozycjach. Sama wysoka aktywacja
nie wystarcza jednak, aby stwierdzić, że filtr rozpoznaje konkretną kategorię
językową. Taką hipotezę trzeba sprawdzić na większej próbce przykładów.
"""


# ============================================================
# Ładowanie modelu
# ============================================================

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

model_args = checkpoint["model_args"]
model = GPT(GPTConfig(**model_args))

state_dict = {
    key.removeprefix("_orig_mod."): value
    for key, value in checkpoint["model"].items()
}

model.load_state_dict(state_dict)
model.eval()

print("Liczba parametrów:", f"{model.get_num_params() / 1e6:.2f}M")

print("\nWarstwy konwolucyjne:")
for name, layer in model.named_modules():
    if isinstance(layer, nn.Conv1d):
        print(name, layer.weight.shape)

if model.cnn is None:
    raise ValueError("Ten checkpoint nie zawiera warstwy CNN.")

conv = model.cnn.conv
n_filters = conv.out_channels

invalid_ids = [
    filter_id for filter_id in filter_ids
    if filter_id < 0 or filter_id >= n_filters
]
if invalid_ids:
    raise ValueError(
        f"Nieprawidłowe indeksy filtrów: {invalid_ids}. "
        f"Model ma {n_filters} filtrów."
    )


# ============================================================
# Tokenizacja i przechwycenie aktywacji
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

text = text.strip()
tokens = tokenizer.encode(text, return_tensors="pt")
original_length = tokens.shape[1]

max_allowed = min(max_tokens, model.config.block_size)
tokens = tokens[:, :max_allowed]

tokens_list = tokens[0].tolist()
token_strings = tokenizer.convert_ids_to_tokens(tokens_list)

# Pomijamy token BOS tylko wtedy, gdy rzeczywiście znajduje się na pozycji 0.
has_bos = (
    tokenizer.bos_token_id is not None
    and tokens_list[0] == tokenizer.bos_token_id
)
start = 1 if has_bos else 0

activations = {}


def save_activation(module, inputs, output):
    # Kształt: (batch, liczba_filtrow, długość_sekwencji)
    activations["cnn.conv"] = output.detach().cpu()


hook = conv.register_forward_hook(save_activation)

try:
    with torch.no_grad():
        model(tokens)
finally:
    hook.remove()

# Kształt: (liczba_filtrow, długość_sekwencji)
activation = activations["cnn.conv"][0]
activation = activation[:, start:]

valid_tokens = token_strings[start:]
valid_token_ids = tokens_list[start:]

# Wybieramy tylko interesujące nas filtry.
selected = activation[filter_ids].numpy()

seq_len = selected.shape[1]

print(f"\nLiczba tokenów przed ograniczeniem: {original_length}")
print(f"Liczba analizowanych tokenów: {seq_len}")
print(f"Analizowane filtry: {filter_ids}")


# ============================================================
# 1. Mapa cieplna: filtry x pozycje tokenów
# ============================================================

# Wspólna skala kolorów ułatwia porównanie odpowiedzi filtrów.
limit = float(np.percentile(np.abs(selected), 99))
if limit == 0:
    limit = 1.0

fig_width = max(12, min(24, seq_len / 35))

fig, ax = plt.subplots(figsize=(fig_width, 6))

image = ax.imshow(
    selected,
    aspect="auto",
    interpolation="nearest",
    cmap="coolwarm",
    vmin=-limit,
    vmax=limit,
)

ax.set_yticks(range(len(filter_ids)))
ax.set_yticklabels(filter_ids)
ax.set_ylabel("Indeks filtra")
ax.set_xlabel("Pozycja tokenu")
ax.set_title("Aktywacje filtrów CNN na kolejnych pozycjach tekstu")

fig.colorbar(image, ax=ax, label="Aktywacja konwolucji")

# Podpisujemy tylko wybrane pozycje, żeby wykres pozostał czytelny.
tick_step = max(1, seq_len // 20)
tick_positions = list(range(0, seq_len, tick_step))

ax.set_xticks(tick_positions)
ax.set_xticklabels(
    [
        f"{pos}: {valid_tokens[pos]}"
        for pos in tick_positions
    ],
    rotation=70,
    ha="right",
    fontsize=8,
)

fig.tight_layout()
fig.savefig(output_dir / "activation_heatmap.png", dpi=180)
plt.close(fig)


# ============================================================
# 2. Wykres aktywacji każdego filtra wzdłuż tekstu
# ============================================================

fig, axes = plt.subplots(
    len(filter_ids),
    1,
    figsize=(14, max(2 * len(filter_ids), 8)),
    sharex=True,
)

if len(filter_ids) == 1:
    axes = [axes]

for ax, filter_id, values in zip(axes, filter_ids, selected):
    ax.plot(np.arange(seq_len), values, linewidth=0.9)
    ax.axhline(0, linewidth=0.6)
    ax.set_ylabel(f"Filtr {filter_id}")
    ax.grid(True, alpha=0.2)

axes[0].set_title("Aktywacja poszczególnych filtrów")
axes[-1].set_xlabel("Pozycja tokenu")

fig.tight_layout()
fig.savefig(output_dir / "activation_lines.png", dpi=180)
plt.close(fig)


# ============================================================
# 3. Wszystkie aktywacje w CSV
# ============================================================

activation_csv = output_dir / "activations_by_token.csv"

with activation_csv.open("w", newline="", encoding="utf-8") as file:
    fields = [
        "position",
        "token_id",
        "token",
        "context",
        *[f"filter_{i}" for i in filter_ids],
    ]

    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    for pos, token in enumerate(valid_tokens):
        # Dla kernel_size=3 uwzględniamy token bieżący i dwa poprzednie.
        context_start = max(0, pos - 2)
        context = tokenizer.convert_tokens_to_string(
            valid_tokens[context_start:pos + 1]
        )

        row = {
            "position": pos,
            "token_id": valid_token_ids[pos],
            "token": token,
            "context": context,
        }

        for row_idx, filter_id in enumerate(filter_ids):
            row[f"filter_{filter_id}"] = float(selected[row_idx, pos])

        writer.writerow(row)


# ============================================================
# 4. Najsilniejsze aktywacje dodatnie i ujemne
# ============================================================

top_csv = output_dir / "strongest_positions.csv"
top_rows = []

for row_idx, filter_id in enumerate(filter_ids):
    values = selected[row_idx]

    positive_positions = np.argsort(values)[::-1][:top_k]
    negative_positions = np.argsort(values)[:top_k]

    for kind, positions in [
        ("najsilniejsze_dodatnie", positive_positions),
        ("najsilniejsze_ujemne", negative_positions),
    ]:
        for pos in positions:
            context_start = max(0, int(pos) - 2)
            context = tokenizer.convert_tokens_to_string(
                valid_tokens[context_start:int(pos) + 1]
            )

            top_rows.append({
                "filter_id": filter_id,
                "typ": kind,
                "pozycja": int(pos),
                "token": valid_tokens[int(pos)],
                "aktywacja": float(values[int(pos)]),
                "kontekst": context,
            })

with top_csv.open("w", newline="", encoding="utf-8") as file:
    fields = [
        "filter_id",
        "typ",
        "pozycja",
        "token",
        "aktywacja",
        "kontekst",
    ]
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(top_rows)


# ============================================================
# 5. Wyniki w terminalu
# ============================================================

print("\nNajsilniejsze aktywacje:")

for filter_id in filter_ids:
    rows = [row for row in top_rows if row["filter_id"] == filter_id]
    print(f"\nFiltr {filter_id}:")

    for kind in ("najsilniejsze_dodatnie", "najsilniejsze_ujemne"):
        matching = [row for row in rows if row["typ"] == kind]

        if matching:
            best = matching[0]
            print(
                f"  {kind}: "
                f"aktywacja={best['aktywacja']:.4f}, "
                f"pozycja={best['pozycja']}, "
                f"token={best['token']!r}, "
                f"kontekst={best['kontekst']!r}"
            )

print("\nZapisano pliki w:", output_dir)
print("  activation_heatmap.png")
print("  activation_lines.png")
print("  activations_by_token.csv")
print("  strongest_positions.csv")

print(
    "\nUwaga: przy kernel_size=3 aktywacja na pozycji t zależy "
    "od wejścia na pozycjach t-2, t-1 i t. Token pokazany w wynikach "
    "oznacza pozycję wyjściową, a niekoniecznie jedyny token "
    "odpowiedzialny za aktywację."
)
