import torch
from torch import nn
from model import GPT, GPTConfig
from transformers import AutoTokenizer
import torch

checkpoint_path = "outs/cnn-transformer-5000-k3.pt"

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

model_args = checkpoint["model_args"]
model = GPT(GPTConfig(**model_args))

state_dict = checkpoint["model"]
state_dict = {
    key.removeprefix("_orig_mod."): value
    for key, value in state_dict.items()
}
model.load_state_dict(state_dict)
model.eval()

print("Warstwy konwolucyjne:")
for name, layer in model.named_modules():
    if isinstance(layer, nn.Conv1d):
        print(name, layer.weight.shape)


activations = {}

def save_activation(name):
    def hook(module, inputs, output):
        activations[name] = output.detach().cpu()
    return hook

model.cnn.conv.register_forward_hook(save_activation("cnn.conv"))

tokenizer = AutoTokenizer.from_pretrained(
    "speakleash/Bielik-1.5B-v3"
)


texts = [
    "Muzyka klasyczna rozwijała się w Europie przez wiele stuleci.",
    "Kot siedział na parapecie i obserwował padający deszcz.",
    "Sieci neuronowe uczą się reprezentacji danych podczas treningu.",
    "Warszawa jest stolicą Polski i największym miastem kraju.",
    "Algorytmy sortowania porządkują elementy według określonego klucza.",
]

for text in texts:
    activations.clear()
    tokens = tokenizer.encode(text, return_tensors="pt")

    with torch.no_grad():
        model(tokens)

    
    activation = activations["cnn.conv"][0]  # (liczba_filtrow, długość_sekwencji)

    tokens_list = tokens[0].tolist()
    token_strings = tokenizer.convert_ids_to_tokens(tokens_list)

    # Ustaw start = 1 tylko wtedy, gdy token 0 jest rzeczywiście BOS.
    start = 1 if tokens_list[0] == tokenizer.bos_token_id else 0

    valid_activation = activation[:, start:]

    # Maksymalna aktywacja dodatnia każdego filtra.
    max_values, max_positions = valid_activation.max(dim=1)

    top_values, top_filters = torch.topk(max_values, k=5)

    print(f"\nTekst: {text}")
    print("Najsilniejsze dodatnie aktywacje:")

    for value, filter_id in zip(
        top_values.tolist(), top_filters.tolist()
    ):
        position = max_positions[filter_id].item() + start
        token = token_strings[position]

        print(
            f"Filtr {filter_id:3} | "
            f"max {value:.4f} | "
            f"pozycja {position:2} | "
            f"token {token!r}"
        )

    # Porównanie średniej aktywacji na początku i w dalszej części tekstu.
    n_positions = valid_activation.shape[1]
    split = max(1, n_positions // 3)

    first_mean = valid_activation[:, :split].mean().item()
    rest_mean = valid_activation[:, split:].mean().item()

    print(f"Średnia aktywacja - pierwsza 1/3: {first_mean:.4f}")
    print(f"Średnia aktywacja - pozostałe 2/3: {rest_mean:.4f}")

