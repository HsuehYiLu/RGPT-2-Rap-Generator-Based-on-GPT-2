from __future__ import annotations
import copy
import glob
import json
import math
import re
import numpy as np
import os
import pronouncing
import random
import shutil
import string as string_utils
import sys
import tempfile
import torch
import torch.optim as optim
import tqdm.notebook as tqdm
import yaml

from hydra import compose
from hydra import initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM
from transformers import GPT2LMHeadModel
from transformers import GPT2Model
from transformers import GPT2Tokenizer

from dataset import merge_lines, reorder, reverse_line
from dataset import VerseDataset
from utils import get_tokenizer

def get_input_ids(
        prompt,
        tokenizer,
        use_bos,
        reverse,
        add_line_token
):
    """
    Arguments:
        prompt: str
        tokenizer: the tokenizer used to generate tokens
        use_bos: bool, use <BOS> token as the beginning of the prompt or not
        reverse: bool, revert the word order or not
        add_line_token: bool, add the <LINE> token at the end of prompt or not
    Return:
        input_ids: torch.LongTensor
    """
    prompt = prompt.strip()
    if add_line_token:
        if prompt != "" and prompt[-6:] != "<LINE>":
            prompt += " <LINE>"
    if use_bos and prompt[:5] != "<BOS>":
        prompt = "<BOS> " + prompt

    if reverse is True:
        input_ids = reverse_line(
            input_ids=tokenizer(prompt, return_tensors="np").input_ids[0],
            use_bos=use_bos,
            tokenizer=tokenizer,
            reverse_last_line=True)
        input_ids = torch.tensor(input_ids).reshape(1, -1)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    return input_ids

def batch_decode(
        outputs,
        tokenizer,
        use_bos,
        reverse,
        reverse_last_line
):
    """
    Arguments:
        outputs: List of torch.LongTensor
        tokenizer: the tokenizer used to decode tokens to words
        use_bos: bool, whether the <BOS> token is used or not
        reverse: bool, whether the tokens are in reverse order or not
    """
    if reverse is True:
        reversed = []
        for output in outputs:
            output = torch.tensor(
                reverse_line(
                    input_ids=output.cpu().numpy(),
                    use_bos=use_bos,
                    tokenizer=tokenizer,
                    reverse_last_line=reverse_last_line)
                ).reshape(-1)
            reversed.append(output)
        outputs = torch.stack(reversed)
    else:
        outputs = torch.stack(outputs)

    outputs = tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=False)

    return outputs

def count_lines(prompt):
    return len(prompt.strip().split("<LINE>")) - 1


def lengths_to_mask(lengths, dtype, device, position="pos"):
    max_len = lengths.max().item()
    if position == "pos":
        mask = torch.arange(
            max_len,
            dtype=lengths.dtype,
            device=lengths.device)
        mask = mask.expand(len(lengths), max_len)
        mask = (mask < lengths.unsqueeze(1))
    else:
        mask = torch.arange(
            max_len - 1, -1, -1,
            dtype=lengths.dtype,
            device=lengths.device)
        mask = mask.expand(len(lengths), max_len)
        mask = (mask < lengths.unsqueeze(1))

    mask = mask.clone().detach()
    mask = mask.to(dtype=dtype, device=device)
    
    return mask

def generate_lines(
        model,
        tokenizer,
        config,
        prompts,
        generate_params,
        num_generation,
        batch_size,
        add_line_token
):
    """
    Generate / finish one line of the limerick. The prompts should be in the 
    correct word order (you don't need to revert the words before passing into
    the function)
    """
    use_bos = config.data.use_bos
    reverse = config.data.reverse
    order = config.data.order

    """
    Step 1:
        concat the input ids into a large tensor; notice that the prompts
        are in variable lengths, thus we need to pad **before** the prompt,
        and generate the attention mask accordingly
    """
    full_input_ids = []
    num_lines = []
    for prompt in prompts:
        num_lines = count_lines(prompt)
        input_ids = get_input_ids(
            prompt=prompt,
            tokenizer=tokenizer,
            use_bos=use_bos,
            reverse=reverse,
            add_line_token=add_line_token)
        input_ids = input_ids.repeat(num_generation, 1)
        full_input_ids.append(input_ids)

    # generate attention mask
    lengths = []
    for input_ids in full_input_ids:
        lengths += [input_ids.shape[1]] * input_ids.shape[0]
    lengths = torch.tensor(lengths, dtype=torch.long)
    full_attention_mask = lengths_to_mask(lengths, torch.long, "cpu", "pre")

    # pad the input ids
    max_seq_len = max([input_ids.shape[1] for input_ids in full_input_ids])
    full_input_ids = [
        torch.cat([
            torch.full(
                (input_ids.shape[0], max_seq_len - input_ids.shape[1]),
                fill_value=tokenizer.eos_token_id, dtype=torch.long
            ),
            input_ids
        ], dim=1)
        for input_ids in full_input_ids]
    full_input_ids = torch.cat(full_input_ids, dim=0)

    num_batches = math.ceil(full_input_ids.shape[0] / batch_size)

    # assume that a line cannot be longer than 30 tokens
    tmp_params = copy.deepcopy(generate_params)
    # if "max_length" in tmp_params:
    #     tmp_params.pop("max_length")
    # tmp_params["max_new_tokens"] = 30

    # Step 2: pass the batch into model to get generation output
    outputs = []
    for i in range(num_batches):
    # for i in tqdm.trange(num_batches, leave=False):
        input_ids = full_input_ids[i * batch_size: (i + 1) * batch_size]
        input_ids = input_ids.to(device=config.device)
        attention_mask = \
            full_attention_mask[i * batch_size: (i + 1) * batch_size]
        attention_mask = attention_mask.to(device=config.device)
        with torch.no_grad():
            output = model.generate(
                input_ids, **tmp_params,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id)
            output = torch.unbind(output)
            outputs.extend(output)
    
    # Step 3: convert the generation result back to strings
    outputs = batch_decode(
        outputs=outputs,
        tokenizer=tokenizer,
        use_bos=use_bos,
        reverse=reverse,
        reverse_last_line=False)

    clean_outputs = []
    for output in outputs:
        new_num_lines = count_lines(output)
        if new_num_lines < num_lines + 1:
            continue
        output = output.strip().split(" <LINE> ")[:num_lines + 1]
        output = " <LINE> ".join(output) + " <LINE>"
        # clean up the prepended tokens
        output = output.replace("<|endoftext|>", "").strip()
        clean_outputs.append(output)
  
    return clean_outputs


    

def finish_lines(
        model,
        tokenizer,
        config,
        prompts,
        generate_params,
        num_generation,
        batch_size
):
    return generate_lines(
        model=model,
        tokenizer=tokenizer,
        config=config,
        prompts=prompts,
        generate_params=generate_params,
        num_generation=num_generation,
        batch_size=batch_size,
        add_line_token=False)

def generate_limericks(
        model,
        tokenizer,
        config,
        prompts,
        generate_params,
        num_generation=10,
        batch_size=1,
        add_line_token=True,
):
    use_bos = config.data.use_bos
    reverse = config.data.reverse
    order = config.data.order

    """
    Step 1:
        concat the input ids into a large tensor; notice that the prompts
        are in variable lengths, thus we need to pad **before** the prompts,
        and generate the attention mask accordingly
    """
    full_input_ids = []
    num_lines = []
    for prompt in prompts:
        num_lines = count_lines(prompt)
        input_ids = get_input_ids(
            prompt=prompt,
            tokenizer=tokenizer,
            use_bos=use_bos,
            reverse=reverse,
            add_line_token=add_line_token)
        input_ids = input_ids.repeat(num_generation, 1)
        full_input_ids.append(input_ids)

    # generate attention mask
    lengths = []
    for input_ids in full_input_ids:
        lengths += [input_ids.shape[1]] * input_ids.shape[0]
    lengths = torch.tensor(lengths, dtype=torch.long)
    full_attention_mask = lengths_to_mask(lengths, torch.long, "cpu", "pre")

    # pad the input ids
    max_seq_len = max([input_ids.shape[1] for input_ids in full_input_ids])
    full_input_ids = [
        torch.cat([
            torch.full(
                (input_ids.shape[0], max_seq_len - input_ids.shape[1]),
                fill_value=tokenizer.eos_token_id, dtype=torch.long
            ),
            input_ids
        ], dim=1)
        for input_ids in full_input_ids]
    full_input_ids = torch.cat(full_input_ids, dim=0)

    num_batches = math.ceil(full_input_ids.shape[0] / batch_size)

    # Step 2: pass the batch into model to get generation output
    outputs = []
    for i in range(num_batches):
    # for i in tqdm.trange(num_batches, leave=False):
        input_ids = full_input_ids[i * batch_size: (i + 1) * batch_size]
        input_ids = input_ids.to(device=config.device)
        attention_mask = \
            full_attention_mask[i * batch_size: (i + 1) * batch_size]
        attention_mask = attention_mask.to(device=config.device)
        with torch.no_grad():
            output = model.generate(
                input_ids, **generate_params,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id)
            output = torch.unbind(output)
            outputs.extend(output)

    # Step 3: convert the generation result back to strings
    outputs = batch_decode(
        outputs=outputs,
        tokenizer=tokenizer,
        use_bos=use_bos,
        reverse=reverse,
        reverse_last_line=False)
    clean_outputs = []

    for output in outputs:
        new_num_lines = count_lines(output)
        num:int = random.choice([8, 16])
        if new_num_lines < num:
            num = new_num_lines
        output = output.strip().split(" <LINE> ")[:num]
        output = " <LINE> ".join(output) + " <LINE>"
        # clean up the prepended tokens
        output = output.replace("<|endoftext|>", "").strip()
        clean_outputs.append(output)

    return clean_outputs

def generate_limericks_two_stage(
        standard_lm,
        reverse_lm,
        standard_tokenizer,
        reverse_tokenizer,
        standard_config,
        reverse_config,
        prompts,
        generate_params,
        num_generation_1=10,
        num_generation_2=1,
        batch_size=64,
):

    first_lines = finish_lines(
        model=standard_lm,
        tokenizer=standard_tokenizer,
        config=standard_config,
        prompts=prompts,
        generate_params=generate_params,
        num_generation=num_generation_1,
        batch_size=batch_size)

    limericks = generate_limericks(
        model=reverse_lm,
        tokenizer=reverse_tokenizer,
        config=reverse_config,
        prompts=first_lines,
        generate_params=generate_params,
        num_generation=num_generation_2,
        batch_size=batch_size)

    return limericks



def pad_tokens(tokens, tokenizer, max_len):
    padded_tokens = [
        tokens_ + [tokenizer.pad_token_id] * (max_len - len(tokens_))
        for tokens_ in tokens]
    attention_mask = [
        [1.] * len(tokens_) + [0.] * (max_len - len(tokens_))
        for tokens_ in tokens]

    padded_tokens = torch.tensor(padded_tokens, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.float)

    return padded_tokens, attention_mask


def load_model(exp_dir, tmp_root=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)), "test")):
    with open(os.path.join(exp_dir, "config.yaml")) as _file:
        config = OmegaConf.create(yaml.safe_load(_file))
    tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(exp_dir, "tokenizer"))

    if not os.path.exists(tmp_root):
        os.makedirs(tmp_root, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=tmp_root)
    states = torch.load(os.path.join(exp_dir, "best-model.ckpt"))
    
    model:GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    model = model.cuda()
    model.load_state_dict(states['model_state_dict'])
    model.save_pretrained(tmp_dir)
    new_model:GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(tmp_dir)
    new_model = new_model.cuda()

    return config, tokenizer, new_model

def main() -> int:

    ## Example of two-stage generation

    standard_exp_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)), "ckpt", "gpt2")
    reverse_exp_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)), "ckpt", "reverse-gpt2")

    standard_config, standard_tokenizer, standard_model = \
        load_model(standard_exp_dir)
    reverse_config, reverse_tokenizer, reverse_model = \
        load_model(reverse_exp_dir)

    generate_params = {
        "do_sample": True,
        "max_length": 100,
    }

    results = []
    for _ in range(500):
        results.append(
            generate_limericks_two_stage(
                standard_lm = standard_model,
                reverse_lm = reverse_model,
                standard_config = standard_config,
                reverse_config = reverse_config,
                standard_tokenizer = standard_tokenizer,
                reverse_tokenizer= reverse_tokenizer,
                prompts = [""],
                generate_params=generate_params,
                num_generation_1=1,
                num_generation_2=1,
                batch_size=1)[0])
    with open(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), "data", "generating", "verses.txt"), "w+") as file:
        for res in results:
            pos_res = re.sub(r'[<][\w]+[>]', '', res)
            file.write(f"{pos_res}\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())