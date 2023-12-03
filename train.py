import torch
import torch.nn as nn
import wandb

from itertools import repeat
from tqdm import tqdm

import config
from data import TinyStoriesDataset, get_tokenizer
from model import LLaMA

def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def generate_square_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def train_epoch(model, criterion, optimizer, scheduler, dataloader, len_epoch, device):
    model.train()
    losses = 0

    for i, (target, target_pad_mask) in enumerate(tqdm(dataloader, desc="train", total=len_epoch)):
        if i == len_epoch:
            break
        target = target.to(device)
        target_pad_mask = target_pad_mask.to(device)[:, :-1]

        target_input = target[:, :-1]

        target_mask = generate_square_mask(target_input.shape[1], device)

        logits = model(target_input, target_mask, target_pad_mask)

        optimizer.zero_grad()

        target_out = target[:, 1:]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), target_out.reshape(-1))
        loss.backward()

        optimizer.step()
        scheduler.step()
        losses += loss.item()

    return losses / len_epoch

def evaluate(model, criterion, dataloader, device):
    model.eval()
    losses = 0

    with torch.no_grad():
        for target, target_pad_mask in tqdm(dataloader, desc="val"):
            target = target.to(device)
            target_pad_mask = target_pad_mask.to(device)[:, :-1]

            target_input = target[:, :-1]

            target_mask = generate_square_mask(target_input.shape[1], device)

            logits = model(target_input, target_mask, target_pad_mask)

            target_out = target[:, 1:]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), target_out.reshape(-1))
            losses += loss.item()

    return losses / len(dataloader)

@torch.no_grad()
def generate(model, tokenizer, examples, device, max_len=200, temperature=1.0):
    model.eval()

    prefix = torch.full((examples, 1), fill_value=tokenizer.bos_id()).to(device)
    for i in range(max_len - 1):
        target_mask = generate_square_mask(prefix.shape[1], device)
        target_pad_mask = (prefix == tokenizer.pad_id()).to(device)

        logits = model(prefix, target_mask, target_pad_mask)
        logits = logits[:, -1, :] / temperature
        next_token = torch.multinomial(logits.softmax(dim=-1), num_samples=1)
        prefix = torch.cat((prefix, next_token), dim=-1)

    return tokenizer.decode_batch(prefix)

def main():
    wandb.init(project="dl2_bhw1", name = config.RUN_NAME)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = get_tokenizer()

    train_dataset = TinyStoriesDataset(config.TRAIN_DIR, config.TRAIN_TXT, tokenizer, train=True)
    train_dataloader = inf_loop(torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4))

    val_dataset = TinyStoriesDataset(config.VAL_DIR, config.VAL_TXT, tokenizer, train=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    model = LLaMA(config.VOCAB_SIZE, config.EMBED_DIM, config.N_HEADS, config.HIDDEN_DIM, config.NUM_LAYERS, config.MAX_SEQ_LEN).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id())

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.LR, steps_per_epoch=config.EPOCH_LEN, epochs=config.EPOCHS, pct_start=0.2)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id())

    for epoch in tqdm(range(config.EPOCHS)):
        train_loss = train_epoch(model, criterion, optimizer, scheduler, train_dataloader, config.EPOCH_LEN, device)
        val_loss = evaluate(model, criterion, val_dataloader, device)
        examples = generate(model, tokenizer, 3, device)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        wandb.log({"examples": [wandb.Text(example) for example in examples]})
        print(f"Epoch {epoch+1} train_loss: {train_loss:.4f} val_loss: {val_loss:.4f}")

        if epoch % config.SAVE_PERIOD == 0:
            torch.save(model.state_dict(), f"model.pth")
    wandb.finish()

if __name__ == "__main__":
    main()