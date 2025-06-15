import torch
import torch.nn.functional as F
from tokenizer import BPETokenizer
from model import KothaTransformer

def sample_logits(logits, top_k=20, top_p=0.95, temperature=1.0):
    logits = logits / temperature
    logits = logits.cpu()
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cum_probs > top_p
    if mask[0]:
        mask[0] = False
    sorted_probs[mask] = 0
    if top_k > 0:
        sorted_probs[top_k:] = 0
    sorted_probs /= sorted_probs.sum()
    next_id = torch.multinomial(sorted_probs, 1).item()
    return sorted_idx[next_id].item()

def chat():
    tok = BPETokenizer()
    tok.load("checkpoints")
    vocab_size = len(tok.idx2token)
    # Try RL model if available, else regular
    import os
    mpath = "checkpoints/kotha-rl.pt" if os.path.exists("checkpoints/kotha-rl.pt") else sorted([f for f in os.listdir("checkpoints") if f.startswith("kotha") and f.endswith(".pt")])[-1]
    model = KothaTransformer(vocab_size)
    model.load_checkpoint(f"checkpoints/{mpath}")
    model.eval()
    context_len = model.context
    print("Kotha: Type your prompt (Bangla or English). Ctrl+C to exit.")
    while True:
        try:
            prompt = input("You: ").strip()
            ids = tok.encode(prompt)
            ids = ids[-context_len:]
            x = torch.LongTensor([ids])
            for _ in range(64):
                logits = model(x)
                logits = logits[0, -1, :]
                next_id = sample_logits(logits, top_k=20, top_p=0.90, temperature=0.9)
                x = torch.cat([x, torch.LongTensor([[next_id]])], dim=1)
                if next_id == tok.token2idx["<eos>"] or x.shape[1] >= context_len:
                    break
            reply = tok.decode(x[0].tolist()[len(ids):])
            print(f"Kotha: {reply.strip()}")
        except KeyboardInterrupt:
            print("\nBye!")
            break

if __name__ == "__main__":
    chat()