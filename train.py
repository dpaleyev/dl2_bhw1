import torch
import torch.nn as nn
import tqdm
import config

def generate_square_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def train_epoch(model, optimizer, criterion, dataloader):
    model.train()
    losses = 0

    for target, target_pad_mask in tqdm(dataloader):
        target = target.to(config.DEVICE)
        target_pad_mask = target_pad_mask.to(config.DEVICE)

        target_input = target[:, :-1]

        target_mask = generate_square_mask(target_input.shape[1], config.DEVICE)

        logits = model(target_input, target_mask, target_pad_mask)

        optimizer.zero_grad()

        target_out = target[:, 1:]
        loss = criterion(logits.reshape(-1), target_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(dataloader)

def train(epochs, model, optimizer, criterion, dataloader):
    for epoch in tqdm(range(epochs)):
        loss = train_epoch(model, optimizer, criterion, dataloader)
        print(f"Epoch {epoch+1} Loss: {loss:.4f}")