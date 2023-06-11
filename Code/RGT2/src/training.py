from __future__ import annotations
import glob
import math
import numpy as np
import os
import random
import shutil
import sys
import omegaconf
import torch
import torch.optim as optim
import tqdm.notebook as tqdm
from hydra import compose
from hydra import initialize_config_dir
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
from transformers import get_scheduler
from dataset import merge_lines, reverse_line
from dataset import VerseDataset
from utils import load_dataset, get_tokenizer

def train_epoch(model, train_loader, optimizer, scheduler, scaler, config):
    model.train()
    optimizer.zero_grad()

    bar = tqdm.tqdm(train_loader, leave=False)
    loss_total = 0.

    for step, batch in enumerate(bar):
        outputs = model(**batch)
        loss = outputs.loss
        loss_total += loss.item()
        loss = loss / config.training.gradient_accumulation_steps
        scaler.scale(loss).backward()
  
        if (
                step % config.training.gradient_accumulation_steps == 0 or
                step == len(train_loader) - 1
        ):
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        bar.set_postfix({"Loss": f"{loss_total / (step + 1):.4f}"})

    return loss_total / len(train_loader)

def validation(model, val_loader):
    model.eval()

    bar = tqdm.tqdm(val_loader, leave=False)
    losses = []

    for step, batch in enumerate(bar):
        with torch.no_grad():
            outputs = model(**batch)

        batch_size = batch['input_ids'].shape[0]
        loss = outputs.loss.item()
        losses.extend([loss for _ in range(batch_size)])

        try:
            perplexity = math.exp(np.mean(losses))
        except OverflowError:
            perplexity = float('inf')

    return perplexity

def main() -> int:
    REVERSE_FLAG:str = int(input("\nDo you want to train the text in reverse?\n1 for yes,  2 for no\n"))
    if REVERSE_FLAG == 2:
        config_path = os.path.join(os.path.dirname(__file__), "config_script")
    elif REVERSE_FLAG == 1:
        config_path = os.path.join(os.path.dirname(__file__), "config_script_reverse")

    # initialize
    if not os.path.exists(config_path):
        os.makedirs(config_path, exist_ok=True)

    initialize_config_dir(config_path)

    # finish configuration
    # change the path to your own shortcut
    config:omegaconf.dictconfig.DictConfig = compose(config_name="config")
    if REVERSE_FLAG == 2:
        config.exp_name = "gpt2"    
        config.data.reverse = False
    elif REVERSE_FLAG == 1:
        config.exp_name = "reverse-gpt2"    
        config.data.reverse = True
    config.data.use_bos = True
    config.data.punctuation = False
    config.training.epochs = 10
    config.training.batch_size = 2

    assert config.exp_name is not None
    print(OmegaConf.to_yaml(config))

    os.makedirs(config.data.ckpt_dir, exist_ok=True)
    exp_dir = f"{config.data.ckpt_dir}/{config.exp_name}"
    os.makedirs(exp_dir, exist_ok=True)
    log_file = f"{exp_dir}/log.txt"

    with open(f"{exp_dir}/config.yaml", 'w') as file:
        file.write(OmegaConf.to_yaml(config))

    # load dataset
    verses_all = load_dataset(config)
    # get customized tokenizer
    tokenizer = get_tokenizer(config)
    tokenizer.save_pretrained(f"{exp_dir}/tokenizer")

    print(f"use_bos: {config.data.use_bos}")
    print(f"reverse: {config.data.reverse}")
    print(f"line order: {config.data.order}")

    # sample = random.sample(verses_all, 1)[0]
    # string = merge_lines(sample, config.data.use_bos, config.data.order)
    # print(f"Lines with separator: {string}")
    # if config.data.reverse:
    #     input_ids = reverse_line(
    #         tokenizer(string)['input_ids'],
    #         use_bos=config.data.use_bos,
    #         tokenizer=tokenizer)
    # else:
    #     input_ids = list(tokenizer(string)['input_ids'])

    # print(f"Tokens: {input_ids}")
    # decoded_string = tokenizer.decode(input_ids)
    # print(f"Decoding result: {decoded_string}")

    np.random.seed(11785)
    random.seed(11785)

    if not config.training.full_train:
        train_data, val_data = train_test_split(verses_all, train_size=0.8)
        if config.debug:
            train_data = train_data[:config.training.batch_size * 8]
            val_data = val_data[:config.training.batch_size * 2]
        print(f"# of training samples: {len(train_data)}")
        print(f"# of validation samples: {len(val_data)}")
    else:
        train_data = verses_all
        if config.debug:
            train_data = train_data[:config.training.batch_size * 8]
        print("NOTE: USE ALL DATA FOR TRAINING")
        print(f"# of training samples: {len(train_data)}")

    train_dataset = VerseDataset(train_data, config, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        drop_last=True,
        shuffle=True,
        collate_fn=train_dataset.gen_collate_fn())

    if not config.training.full_train:
        val_dataset = VerseDataset(val_data, config, tokenizer)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            drop_last=False,
            shuffle=False,
            collate_fn=val_dataset.gen_collate_fn())
    else:
        val_dataset, val_loader = None, None

    # initialize the model, also resize the embeddings for new tokens
    model:GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    model = model.cuda()

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)],
            "weight_decay": config.training.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.training.learning_rate)

    T_epoch = np.ceil(
        len(train_loader) //
        config.training.gradient_accumulation_steps)

    scheduler = get_scheduler(
        name=config.training.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.training.num_warmup_steps,
        num_training_steps=config.training.epochs * T_epoch)
    scaler = torch.cuda.amp.grad_scaler.GradScaler()

    files = glob.glob(f"{exp_dir}/epoch-*.ckpt")
    if len(files) != 0:
        files = sorted(files, key=lambda x: int(os.path.basename(x)[6:-5]))
        states = torch.load(files[-1])
        
        model.load_state_dict(states['model_state_dict'])
        optimizer.load_state_dict(states['optimizer_state_dict'])
        scheduler.load_state_dict(states['scheduler_state_dict'])
        scaler.load_state_dict(states['scaler_state_dict'])
        start_epoch = states['epoch'] + 1
        best_perplexity = states['perplexity']
    else:
        start_epoch = 0
        if config.training.full_train:
            best_perplexity = 0
        else:
            best_perplexity = 1e30

    if start_epoch == 0:
        print("Start training from scratch")
    else:
        print(f"Resume training from epoch {start_epoch + 1}")

    epoch_bar = tqdm.trange(start_epoch, config.training.epochs, leave=False)

    for epoch in epoch_bar:
        loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, config)
        flag = False

        if config.training.full_train:
            perplexity = 0
            log = f"Epoch {epoch+1} Loss: {loss:.4f}"
        else:
            perplexity = validation(model, val_loader)
            log = f"Epoch {epoch+1} Loss: {loss:.4f} Perplexity {perplexity:.4f}"
        
            if perplexity < best_perplexity:
                best_perplexity = perplexity
                flag = True

        epoch_bar.write(log)
        with open(log_file, 'a') as file:
            file.write(f"{log}\n")

        epoch_bar.write(f"Save model at epoch {epoch+1}")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': 
                scheduler.state_dict()
                if scheduler is not None else None,
            'scaler_state_dict': scaler.state_dict(),
            'epoch': epoch,
            'perplexity': perplexity,
            'best_perplexity': best_perplexity
        }, f"{exp_dir}/epoch-{epoch+1}.ckpt")
        if epoch != 0:
            prev_ckpt = f"{exp_dir}/epoch-{epoch}.ckpt"
            if os.path.exists(prev_ckpt):
                os.remove(f"{exp_dir}/epoch-{epoch}.ckpt")

        if flag or config.training.full_train:
            print(f"Save best model at epoch {epoch+1}")
            best_perplexity = perplexity
            shutil.copyfile(
                f"{exp_dir}/epoch-{epoch+1}.ckpt",
                f"{exp_dir}/best-model.ckpt")
    return 0

if __name__ == "__main__":
    sys.exit(main())



