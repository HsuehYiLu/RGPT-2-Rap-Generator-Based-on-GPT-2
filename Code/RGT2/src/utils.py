from __future__ import annotations
import json
from transformers import AutoTokenizer, GPT2Tokenizer
import omegaconf
import os



def get_tokenizer(config:omegaconf.dictconfig.DictConfig) -> GPT2Tokenizer:
    """cutomize tokenizer

    Args:
        config (omegaconf.dictconfig.DictConfig): configuration

    Returns:
        GPT2Tokenizer: customized tokenizer
    """

    tokenizer:GPT2Tokenizer = AutoTokenizer.from_pretrained("gpt2")
    special_tokens = {
        "sep_token": "<LINE>",
        "pad_token": "<PAD>",
        "bos_token": "<BOS>"
    }

    if not config.data.use_bos:
        special_tokens.pop("bos_token")

    tokenizer.add_special_tokens(special_tokens)

    for key in special_tokens:
        print(key)
        print(
            f"New {key}: {getattr(tokenizer, key)} "
            f"({getattr(tokenizer, key + '_id')})")

    return tokenizer


def load_dataset(config:omegaconf.dictconfig.DictConfig) -> list[list[str]]:
    """load dataset from saved json file

    Args:
        config (omegaconf.dictconfig.DictConfig): configuration

    Returns:
        list[list[str]]: list of lists(verses)
    """
    with open(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), "data", "preprocessing", "verses.json")) as file:
        data = json.load(file)
    verses_all:list[list[str]] = []

    for _, _verses in data['verses'].items():
        verses = _verses['verse']
        
        verses_all.append(verses)

    print(f"# of verses before clean-up: {len(data['verses'])}")
    print(f"# of verses after clean-up: {len(verses_all)}")

    return verses_all