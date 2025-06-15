import regex as re
import json
import argparse
from collections import Counter, defaultdict
from tqdm import tqdm

class BPETokenizer:
    def __init__(self, vocab_size=8000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.idx2token = []
        self.token2idx = {}
        self.bpe_merges = []
        self.special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

    def is_bangla(self, c):
        return "\u0980" <= c <= "\u09FF"

    def tokenize_line(self, line):
        # Split into words, keep Bangla and English
        # Use regex to separate punctuation
        pattern = r"[\p{Bengali}\w]+|[^\s\p{Bengali}\w]"
        return re.findall(pattern, line, flags=re.UNICODE)

    def get_stats(self, corpus):
        """Count pairs of symbols in tokenized corpus."""
        pairs = Counter()
        for word in corpus:
            prev_char = word[0]
            for char in word[1:]:
                pairs[(prev_char, char)] += 1
                prev_char = char
        return pairs

    def merge_vocab(self, pair, corpus):
        """Merge the most frequent pair in the corpus."""
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        new_corpus = []
        for word in corpus:
            word_str = " ".join(word)
            word_str = pattern.sub("".join(pair), word_str)
            new_corpus.append(word_str.split(" "))
        return new_corpus

    def train(self, data_path):
        # Step 1: Build initial vocab (char-level, with special tokens)
        corpus = []
        word_freq = Counter()
        with open(data_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Reading data"):
                words = self.tokenize_line(line.strip())
                for w in words:
                    word_freq[w] += 1
        for word, freq in word_freq.items():
            corpus.extend([list(word) + ["</w>"]] * freq)
        vocab = Counter([" ".join(word) for word in corpus])

        # Step 2: BPE merges
        bpe_merges = []
        for _ in tqdm(range(self.vocab_size - len(self.special_tokens)), desc="BPE merges"):
            pairs = self.get_stats(corpus)
            if not pairs: break
            best = max(pairs, key=pairs.get)
            bpe_merges.append(best)
            corpus = self.merge_vocab(best, corpus)

        # Step 3: Build final vocab
        tokens = set()
        for word in corpus:
            tokens.update(word)
        self.idx2token = self.special_tokens + sorted(tokens)
        self.token2idx = {t: i for i, t in enumerate(self.idx2token)}
        self.bpe_merges = bpe_merges

    def save(self, out_dir="checkpoints"):
        with open(f"{out_dir}/vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.idx2token, f, ensure_ascii=False, indent=2)
        with open(f"{out_dir}/tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump({"bpe_merges": self.bpe_merges, "special_tokens": self.special_tokens}, f, ensure_ascii=False, indent=2)

    def load(self, out_dir="checkpoints"):
        with open(f"{out_dir}/vocab.json", "r", encoding="utf-8") as f:
            self.idx2token = json.load(f)
        self.token2idx = {t: i for i, t in enumerate(self.idx2token)}
        with open(f"{out_dir}/tokenizer_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
            self.bpe_merges = cfg["bpe_merges"]
            self.special_tokens = cfg["special_tokens"]

    def encode_word(self, word):
        # BPE encoding
        chars = list(word) + ["</w>"]
        pairs = [(chars[i], chars[i+1]) for i in range(len(chars)-1)]
        merges = {pair: i for i, pair in enumerate(self.bpe_merges)}
        while True:
            pair_set = set((chars[i], chars[i+1]) for i in range(len(chars)-1))
            candidates = [p for p in pair_set if p in merges]
            if not candidates: break
            best = min(candidates, key=lambda p: merges[p])
            i = 0
            new_chars = []
            while i < len(chars):
                if i < len(chars)-1 and (chars[i], chars[i+1]) == best:
                    new_chars.append(chars[i]+chars[i+1])
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
        return [c for c in chars if c != "</w>"]

    def encode(self, text, add_special=True):
        tokens = self.tokenize_line(text.strip())
        ids = []
        if add_special:
            ids.append(self.token2idx.get("<bos>", 1))
        for t in tokens:
            bpe_tokens = self.encode_word(t)
            for bpe in bpe_tokens:
                ids.append(self.token2idx.get(bpe, self.token2idx.get("<unk>", 1)))
        if add_special:
            ids.append(self.token2idx.get("<eos>", 1))
        return ids

    def decode(self, ids, skip_special=True):
        tokens = []
        for i in ids:
            t = self.idx2token[i]
            if skip_special and t in self.special_tokens: continue
            tokens.append(t)
        # Merge BPE tokens into words
        text = "".join(tokens)
        return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help="Train BPE tokenizer with data.txt")
    parser.add_argument('--vocab_size', type=int, default=8000)
    args = parser.parse_args()
    tok = BPETokenizer(vocab_size=args.vocab_size)
    if args.train:
        tok.train(args.train)
        tok.save("checkpoints")
        print("Tokenizer trained and saved to checkpoints/")
    else:
        print("Usage: python tokenizer.py --train data.txt")