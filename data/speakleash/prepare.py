# prepares the Speakleash dataset for training, using a compact flow similar to openwebtext

import os
import pickle
import random

import numpy as np
from speakleash import Speakleash
from tqdm import tqdm
from transformers import AutoTokenizer


TOKENIZER_NAME = "speakleash/Bielik-1.5B-v3"
DATASET_NAME = "plwiki"
MAX_DOCS = 100000
MIN_TEXT_LENGTH = 150
SPLIT_FRACTION = 0.995


def extract_text(doc):
    if isinstance(doc, str):
        return doc.strip()
    if isinstance(doc, dict):
        for key in ("text", "content", "document"):
            value = doc.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        parts = [value.strip() for value in doc.values() if isinstance(value, str) and value.strip()]
        return "\n".join(parts).strip()
    return str(doc).strip()


def main():
    print(f"Downloading tokenizer: {TOKENIZER_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")

    if vocab_size > np.iinfo(np.uint16).max:
        raise ValueError(
            f"tokenizer vocab_size={vocab_size} does not fit into uint16; "
            "update the training loader to uint32 before preparing the dataset"
        )

    dtype = np.uint16

    cache_dir = os.environ.get("SPEAKLEASH_CACHE_DIR")
    if not cache_dir:
        raise EnvironmentError("SPEAKLEASH_CACHE_DIR is not set")
    if cache_dir.startswith(os.path.expanduser("~")):
        raise EnvironmentError(
            "SPEAKLEASH_CACHE_DIR should not point inside your home directory"
        )

    sl = Speakleash(cache_dir)

    print(f"Downloading speakleash dataset: {DATASET_NAME}...")
    dset = sl.get(DATASET_NAME)

    print(f"Extracting up to {MAX_DOCS} documents...")
    documents = []
    for i, doc in enumerate(tqdm(dset.data, total=MAX_DOCS)):
        if i >= MAX_DOCS:
            break
        text = extract_text(doc)
        if len(text) > MIN_TEXT_LENGTH:
            documents.append(text)

    print(f"Downloaded {len(documents)} documents")

    random.seed(42)
    random.shuffle(documents)

    split_idx = int(len(documents) * SPLIT_FRACTION)
    splits = {
        "train": documents[:split_idx],
        "val": documents[split_idx:],
    }

    for split_name, docs in splits.items():
        filename = os.path.join(os.path.dirname(__file__), f"{split_name}.bin")
        print(f"Tokenizing and writing {split_name} split to {filename}...")

        token_count = 0
        with open(filename, "wb") as f:
            for text in tqdm(docs, desc=f"writing {filename}"):
                ids = tokenizer(text, add_special_tokens=False).input_ids
                if tokenizer.eos_token_id is not None:
                    ids.append(tokenizer.eos_token_id)
                np.asarray(ids, dtype=dtype).tofile(f)
                token_count += len(ids)

        size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"Done! {filename} ({token_count:,} tokens, {size_mb:.2f} MB)")

    meta = {
        "vocab_size": vocab_size,
        "tokenizer_name": TOKENIZER_NAME,
        "dtype": np.dtype(dtype).name,
        "dataset_name": DATASET_NAME,
        "max_docs": MAX_DOCS,
        "min_text_length": MIN_TEXT_LENGTH,
        "split_fraction": SPLIT_FRACTION,
    }
    with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print("Done! meta.pkl saved.")


if __name__ == "__main__":
    main()
