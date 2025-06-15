# Kotha LLM

Kotha is a **from-scratch, CPU-only bilingual Large Language Model (LLM)** designed for Bangla and English, trained on user-provided data with a custom tokenizer and transformer—all running in a Python virtual environment on Windows 11.

---

## 🚀 Features

- **Custom BPE Tokenizer:** Handles Bangla (Unicode) & English from raw text.
- **PyTorch Transformer:** 4-layer, 256-d hidden size, causal masking, CPU-optimized.
- **No Pretrained Models:** All components built from scratch, no external LLMs or tokenizers.
- **Full Training Pipeline:** Tokenization, supervised training, RL fine-tuning (REINFORCE), checkpointing.
- **Interactive CLI:** Chat with the model in Bangla, English, or both.
- **Runs on CPU:** No GPU or CUDA required.
- **Single-Machine Friendly:** Designed for 8 GB RAM, 4-core CPU.

---

## 🖥️ Host Requirements

| Spec   | Value                     |
| ------ | ------------------------- |
| OS     | Windows 11 (64-bit)       |
| CPU    | Intel Core i5-4590 (4c)   |
| RAM    | 8 GB                      |
| GPU    | None (CPU only)           |
| Python | ≥3.9                      |

---

## 📂 Project Layout

```
kotha-llm/
├── data.txt                 # Training corpus (Bangla + English)
├── setup_virtualenv.bat     # Virtualenv installer (Windows)
├── tokenizer.py             # Custom tokenizer
├── model.py                 # Transformer model
├── train.py                 # Training loop
├── rl_trainer.py            # RL (REINFORCE) fine-tuner
├── inference.py             # Inference CLI
└── checkpoints/             # Saved models/tokenizer
```

---

## ⚙️ Setup

1. **Clone this repo** and add your `data.txt` (~1 GB Bangla+English text corpus).
2. **Install Python 3.9+** (from [python.org](https://www.python.org/downloads/windows/)).
3. **Run setup:**
    ```bat
    setup_virtualenv.bat
    ```
4. **Train Tokenizer:**
    ```bash
    python tokenizer.py --train data.txt
    ```
5. **Train Base Model:**
    ```bash
    python train.py
    ```
6. **RL Fine-tune (optional):**
    ```bash
    python rl_trainer.py
    ```
7. **Start Chat:**
    ```bash
    python inference.py
    ```

---

## 🧩 Components

### 1. Tokenizer (`tokenizer.py`)
- Learns BPE or WordPiece vocab from `data.txt`
- Handles Unicode (Bangla) and English text
- Saves `vocab.json`, `tokenizer_config.json`

### 2. Model (`model.py`)
- PyTorch Transformer:
  - 4 layers, 256 hidden, 4 heads, context 128
  - Causal masking, dropout 0.1
- Save/load via checkpoints

### 3. Training (`train.py`)
- Reads & tokenizes `data.txt`
- Implements gradient accumulation (big batch on CPU)
- Prints live loss, perplexity
- Saves checkpoints

### 4. RL Fine-Tuning (`rl_trainer.py`)
- Loads model/tokenizer
- REINFORCE on sampled generations
- Reward: grammar, Bangla/English fluency
- Penalty: repetition, hallucination
- Saves `kotha-rl.pt`

### 5. Inference CLI (`inference.py`)
- Loads model/tokenizer
- User prompt → sampled reply
- Supports:
  - Top-k / top-p sampling
  - Temperature
  - Bangla/English mix

---

## 🧪 Example

```bash
> python inference.py
You: তোমার নাম কি?
Kotha: আমার নাম কথা। আমি বাংলা ও ইংরেজি বুঝি।
You: What's the capital of Bangladesh?
Kotha: The capital of Bangladesh is Dhaka.
```

---

## ⚡ Performance

| Task                | Value              |
| ------------------- | ------------------ |
| Tokenizer Training  | 2–5 mins           |
| Base Model Training | ~8–14 hours (CPU)  |
| RL Fine-Tuning      | ~1–2 hours         |
| Inference           | < 1s per response  |
| RAM Used            | ~6–7 GB peak       |

---

## 🛑 Restrictions

- **NO HuggingFace, GPT, Llama, or other pretrained LLMs**
- **NO CUDA/GPU**
- **NO external APIs or cloud**
- **All computation local and CPU-only**

---

## 📦 Python Dependencies

Install via `setup_virtualenv.bat`:

```
pip install torch numpy tqdm regex
```

Optional:

```
pip install matplotlib rich
```

---

## 🧠 FAQ

**Q: Can I use this on Linux or Mac?**  
A: The scripts are written for Windows but can be ported with minor changes.

**Q: Can I use my own data?**  
A: Yes! Replace `data.txt` with your own bilingual corpus.

**Q: Will this run on just 8 GB RAM?**  
A: Yes, if you keep batch/context sizes moderate.

---

## ✨ Credits

- Inspired by GPT/Transformer papers, but **100% original code**
- Unicode/Bangla support from scratch

---

## 📬 Contact

For issues or contributions, open an issue or PR on GitHub.
