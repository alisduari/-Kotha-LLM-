import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
from tokenizer import BPETokenizer
from model import KothaTransformer

BATCH_SIZE = 16
CONTEXT = 128
EPOCHS = 1  # For demo; increase for better results
LR = 3e-4
ACCUM_STEPS = 4

def batchify(data, batch_size, context):
    # Returns batches of shape (batch, context)
    n_batches = len(data) // (batch_size * context)
    data = data[:n_batches * batch_size * context]
    data = np.array(data).reshape(batch_size, -1)
    for i in range(0, data.shape[1] - context, context):
        x = data[:, i:i+context]
        y = data[:, i+1:i+context+1]
        yield torch.LongTensor(x), torch.LongTensor(y)

def main():
    os.makedirs("checkpoints", exist_ok=True)
    # 1. Load tokenizer
    tok = BPETokenizer()
    tok.load("checkpoints")
    vocab_size = len(tok.idx2token)
    # 2. Load data and encode
    encoded = []
    with open("data.txt", "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Encoding data"):
            encoded += tok.encode(line)
    print(f"Total tokens: {len(encoded)}")

    # 3. Model
    device = torch.device("cpu")
    model = KothaTransformer(vocab_size=vocab_size, n_layers=4, d_model=256, n_heads=4, context=CONTEXT)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=tok.token2idx["<pad>"])

    step = 0
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}")
        losses = []
        for x, y in tqdm(batchify(encoded, BATCH_SIZE, CONTEXT), desc="Training"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            (loss / ACCUM_STEPS).backward()
            if (step + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            losses.append(loss.item())
            if step % 100 == 0:
                ppl = np.exp(np.mean(losses[-100:]))
                print(f"Step {step}, Loss: {loss.item():.4f}, PPL: {ppl:.2f}")
            if step % 1000 == 0:
                model.save_checkpoint(f"checkpoints/kotha_step{step}.pt")
            step += 1
        # Save after each epoch
        model.save_checkpoint(f"checkpoints/kotha_epoch{epoch+1}.pt")

if __name__ == "__main__":
    main()