import torch
import torch.optim as optim
import numpy as np
from tokenizer import BPETokenizer
from model import KothaTransformer
from tqdm import tqdm
import random

def reward_fn(text):
    # Simple reward: +1 if contains Bangla or English, -1 if repetitive or nonsense (very naive!)
    if len(set(text.split())) < 3:  # Repetition penalty
        return -1
    if any('\u0980' <= c <= '\u09FF' for c in text) or any('a' <= c.lower() <= 'z' for c in text):
        return 1
    return -0.5

def sample(model, tok, prompt, max_len=64, temperature=1.0):
    model.eval()
    ids = tok.encode(prompt)
    ids = ids[:model.context]
    x = torch.LongTensor([ids]).to('cpu')
    for _ in range(max_len):
        logits = model(x)
        logits = logits[0, -1, :] / temperature
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        x = torch.cat([x, torch.LongTensor([[next_id]]).to('cpu')], dim=1)
        if next_id == tok.token2idx["<eos>"]:
            break
    return tok.decode(x[0].tolist())

def main():
    # Load
    tok = BPETokenizer()
    tok.load("checkpoints")
    vocab_size = len(tok.idx2token)
    model = KothaTransformer(vocab_size)
    model.load_checkpoint("checkpoints/kotha_epoch1.pt")
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    # RL loop
    EPISODES = 100
    for ep in tqdm(range(EPISODES), desc="RL episodes"):
        prompt = random.choice([
            "তুমি কেমন আছ?",
            "What is the capital of Bangladesh?",
            "তোমার নাম কি?",
            "Who are you?",
            "বাংলা ভাষা কেমন?",
            "Tell me a joke."
        ])
        out = sample(model, tok, prompt, max_len=32)
        R = reward_fn(out)
        # Policy gradient loss (REINFORCE, naive)
        # Use log-prob of generated tokens
        ids = tok.encode(prompt)
        x = torch.LongTensor([ids]).to('cpu')
        model.zero_grad()
        logits = model(x)
        logprobs = torch.nn.functional.log_softmax(logits[0, :-1, :], dim=-1)
        chosen = x[0, 1:]
        lp = logprobs[range(len(chosen)), chosen]
        loss = -lp.mean() * R
        loss.backward()
        optimizer.step()
        if ep % 10 == 0:
            print(f"EP {ep}: Prompt: {prompt} | Reply: {out} | Reward: {R}")
    model.save_checkpoint("checkpoints/kotha-rl.pt")
    print("RL fine-tuned model saved to checkpoints/kotha-rl.pt")

if __name__ == "__main__":
    main()