
import torch
from model import GPT, GPTConfig
from transformers import AutoTokenizer

checkpoint_path = "outs/cnn-transformer-5000-k3.pt"
tokenizer_name = "speakleash/Bielik-1.5B-v3"

FILTER_ID = 340
TOP_K = 10
MAX_TOKENS = 800
CONTEXT_RADIUS = 2


# ============================================================
# Ładowanie modelu
# ============================================================

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

model = GPT(GPTConfig(**checkpoint["model_args"]))

state_dict = {
    key.removeprefix("_orig_mod."): value
    for key, value in checkpoint["model"].items()
}

model.load_state_dict(state_dict)
model.eval()

if model.cnn is None:
    raise ValueError("Model nie zawiera warstwy CNN.")

if FILTER_ID >= model.cnn.conv.out_channels:
    raise ValueError(f"Model nie ma filtra {FILTER_ID}.")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


# ============================================================
# Tekst do analizy
# ============================================================

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
# Tokenizacja
# ============================================================

tokens = tokenizer.encode(text, return_tensors="pt")
tokens = tokens[:, :min(MAX_TOKENS, model.config.block_size)]

token_ids = tokens[0].tolist()
token_strings = tokenizer.convert_ids_to_tokens(token_ids)

# Pomijamy BOS wyłącznie wtedy, gdy rzeczywiście jest na pozycji 0.
has_bos = (
    tokenizer.bos_token_id is not None
    and token_ids[0] == tokenizer.bos_token_id
)
start = 1 if has_bos else 0


# ============================================================
# Przechwycenie aktywacji filtra 340
# ============================================================

activations = {}

def hook_fn(module, inputs, output):
    # output: (batch, liczba_filtrow, długość_sekwencji)
    activations["conv"] = output.detach().cpu()

hook = model.cnn.conv.register_forward_hook(hook_fn)

try:
    with torch.no_grad():
        model(tokens)
finally:
    hook.remove()

# Aktywacje filtra 340 na kolejnych pozycjach.
values = activations["conv"][0, FILTER_ID, start:].tolist()

valid_tokens = token_strings[start:]
valid_ids = token_ids[start:]


# ============================================================
# TOP 10 dodatnich aktywacji
# ============================================================

top_positions = sorted(
    range(len(values)),
    key=lambda pos: values[pos],
    reverse=True,
)[:TOP_K]

print(f"\nTOP {TOP_K} aktywacji filtra {FILTER_ID}\n")

for rank, pos in enumerate(top_positions, start=1):
    left = max(0, pos - CONTEXT_RADIUS)
    right = min(len(valid_tokens), pos + CONTEXT_RADIUS + 1)

    # Kontekst wyświetlany jako czytelny tekst.
    context = tokenizer.decode(
        valid_ids[left:right],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    # Tokeny faktycznie używane przez konwolucję przy kernel_size=3.
    conv_left = max(0, pos - 2)
    conv_context = tokenizer.decode(
        valid_ids[conv_left:pos + 1],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    print(f"{rank}. Aktywacja: {values[pos]:.4f}")
    print(f"   Pozycja: {pos}")
    print(f"   Token na pozycji aktywacji: {valid_tokens[pos]!r}")
    print(f"   Okno 5 tokenów: {context!r}")
    print(f"   Wejście konwolucji (do 3 tokenów): {conv_context!r}")
    print()
