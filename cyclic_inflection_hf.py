# Honestly, made with Claude Code by asking it to refactor my previous script,
# to make use of data batching.
import itertools
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime as dt
from tqdm import tqdm

# --- HuggingFace imports ---
from datasets import Dataset
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
SAMPLE_RATE = 0.04
EARLY_STOP_THRESHOLD = 10
FORCED_PRETRAIN_EPOCHS = 1
GET_ACCURACY_EPOCHS = 10
BATCH_SIZE = 32  # Used for DataLoader batching; pairs are still built inside collate
LANG = "eng"

# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────

def load_split(path: str, sample_rate: float, add_root_rows: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["root", "content", "form"])
    df = df.sample(frac=sample_rate, random_state=42)
    if add_root_rows:
        root_rows = pd.DataFrame([
            {"root": r, "content": "ROOT", "form": f"*{r}"}
            for r in df.root.unique()
        ])
        df = pd.concat([df, root_rows], ignore_index=True)
    return df


train_df = load_split(f"part1/data/{LANG}.trn", SAMPLE_RATE, add_root_rows=True)
test_df  = load_split(f"part1/data/{LANG}.tst", SAMPLE_RATE, add_root_rows=False)
dev_df   = load_split(f"part1/data/{LANG}.dev", SAMPLE_RATE, add_root_rows=True)

print(f"Num train rows: {train_df.shape[0]}")
print(f"Num test rows:  {test_df.shape[0]}")
print(f"Num dev rows:   {dev_df.shape[0]}")

# ─────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────
GLOSS_START, GLOSS_END = "GLOSS_START", "GLOSS_END"

form_unigram_vocab: set = set()
for row in train_df.itertuples():
    form_unigram_vocab |= set(row.form)
specials = {GLOSS_START, GLOSS_END, "OOV", "PAD"}
if len(form_unigram_vocab | specials) % 2:
    specials.add("FILL")
form_unigram_vocab |= specials
form_unigram_vocab: dict = {w: i for i, w in enumerate(form_unigram_vocab)}

content_unigram_vocab: set = set()
for row in train_df.itertuples():
    content_unigram_vocab |= set(row.content.split(";"))
content_specials = {"OOV"}
if len(content_unigram_vocab | content_specials) % 2:
    content_specials.add("FILL")
content_unigram_vocab |= content_specials
content_unigram_vocab: dict = {w: i for i, w in enumerate(content_unigram_vocab)}

print(f"Form vocab size:    {len(form_unigram_vocab)}")
print(f"Content vocab size: {len(content_unigram_vocab)}")

d_form    = len(form_unigram_vocab)
d_content = len(content_unigram_vocab)

# ─────────────────────────────────────────────
# Tensor encoding helpers
# ─────────────────────────────────────────────

def get_form_sequence(word: str) -> torch.Tensor:
    """Returns (seq_len, d_form) one-hot tensor, wrapped with START/END tokens."""
    idxs = [form_unigram_vocab.get(c, form_unigram_vocab["OOV"]) for c in word]
    idxs = [form_unigram_vocab[GLOSS_START]] + idxs + [form_unigram_vocab[GLOSS_END]]
    return torch.stack([
        torch.Tensor([1.0 if i == j else 0.0 for i in range(d_form)])
        for j in idxs
    ])


def get_content_tensor(content: str) -> torch.Tensor:
    """Returns (d_content,) multi-hot tensor."""
    idxs = {content_unigram_vocab.get(c, content_unigram_vocab["OOV"])
            for c in content.split(";")}
    return torch.Tensor([1.0 if i in idxs else 0.0 for i in range(d_content)])


# Shared pad / start vectors (will be moved to device later)
pad_vector   = torch.zeros(1, d_form); pad_vector[0, form_unigram_vocab["PAD"]] = 1.0
start_vector = torch.zeros(1, d_form); start_vector[0, form_unigram_vocab[GLOSS_START]] = 1.0

# ─────────────────────────────────────────────
# Build HuggingFace Datasets
# ─────────────────────────────────────────────

def df_to_hf_dataset(df: pd.DataFrame) -> Dataset:
    """Convert a pandas DataFrame to a HuggingFace Dataset with pre-encoded tensors."""
    records = []
    for row in df.itertuples():
        records.append({
            "root":           row.root,
            "content":        row.content,
            "form":           row.form,
            # Store tensors as lists so HF can serialise them
            "content_tensor": get_content_tensor(row.content).tolist(),
            "form_sequence":  get_form_sequence(row.form).tolist(),
        })
    return Dataset.from_list(records)


train_dataset = df_to_hf_dataset(train_df)
dev_dataset   = df_to_hf_dataset(dev_df)
test_dataset  = df_to_hf_dataset(test_df)

# Build a root → list-of-indices lookup (used during collation)
def build_root_index(dataset: Dataset) -> dict:
    index: dict[str, list[int]] = {}
    for i, root in enumerate(dataset["root"]):
        index.setdefault(root, []).append(i)
    return index

train_root_index = build_root_index(train_dataset)
dev_root_index   = build_root_index(dev_dataset)

# ─────────────────────────────────────────────
# Collate: expand each sample into (row, pair) tuples
# ─────────────────────────────────────────────

def make_collate_fn(dataset: Dataset, root_index: dict):
    """
    For each item in the batch, sample ONE random pair from the same root.
    Returns a list of (row_dict, pair_dict) tuples – one per original item.
    """
    def collate_fn(batch):
        pairs = []
        for item in batch:
            root = item["root"]
            candidates = root_index[root]
            pair_idx = candidates[torch.randint(len(candidates), (1,)).item()]
            pairs.append((item, dataset[pair_idx]))
        return pairs
    return collate_fn


def make_dataloader(dataset: Dataset, root_index: dict, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=make_collate_fn(dataset, root_index),
        num_workers=0,  # Set >0 if your environment supports forking
        pin_memory=(device.type == "cuda"),
    )

# ─────────────────────────────────────────────
# Models  (unchanged architecture, moved to GPU-friendly init)
# ─────────────────────────────────────────────

class FormModel(nn.Module):
    def __init__(self, d_input: int, num_glosses: int, max_len: int = 50):
        super().__init__()
        self.conv1 = nn.Conv1d(d_input,     256,         kernel_size=3, padding="same")
        self.conv2 = nn.Conv1d(256,         num_glosses, kernel_size=3, padding="same")
        self.transformer = nn.Transformer(
            d_model=num_glosses, nhead=2,
            num_encoder_layers=1, num_decoder_layers=1,
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)

        pos = torch.zeros(max_len, num_glosses)
        positions = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        div = torch.exp(torch.arange(0, num_glosses, 2).float() * (-math.log(10000.0)) / num_glosses)
        pos[:, 0::2] = torch.sin(positions * div)
        pos[:, 1::2] = torch.cos(positions * div)
        self.register_buffer("pos_encoding", pos)

    def forward(self, X, y, tgt_mask=None, tgt_is_causal=False):
        X = self.conv1(X.unsqueeze(0).permute(0, 2, 1))   # (1, C, L)
        X = self.conv2(X).squeeze(0).permute(1, 0)         # (L, C)
        X = X + self.pos_encoding[:X.shape[0]]
        out = self.transformer(X, y, tgt_mask=tgt_mask, tgt_is_causal=tgt_is_causal)
        return self.log_softmax(out)


class ContentModel(nn.Module):
    def __init__(self, d_input: int, num_labels: int, max_len: int = 50):
        super().__init__()
        self.num_labels = num_labels
        self.conv1 = nn.Conv1d(d_input,          256,             kernel_size=3, padding="same")
        self.conv2 = nn.Conv1d(256,              2 * num_labels,  kernel_size=3, padding="same")
        self.transformer = nn.Transformer(
            d_model=2 * num_labels, nhead=2,
            num_encoder_layers=1, num_decoder_layers=1,
        )
        self.log_softmax = nn.LogSoftmax(dim=0)

        pos = torch.zeros(max_len, 2 * num_labels)
        positions = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        div = torch.exp(torch.arange(0, 2 * num_labels, 2).float() * (-math.log(10000.0)) / (2 * num_labels))
        pos[:, 0::2] = torch.sin(positions * div)
        pos[:, 1::2] = torch.cos(positions * div)
        self.register_buffer("pos_encoding", pos)
        self.register_buffer("start_token", torch.zeros(1, 2 * num_labels))

    def forward(self, X):
        X = self.conv1(X.unsqueeze(0).permute(0, 2, 1))   # (1, C, L)
        X = self.conv2(X).squeeze(0).permute(1, 0)         # (L, C)
        X = X + self.pos_encoding[:X.shape[0]]
        out = self.transformer(X, self.start_token).reshape(2, self.num_labels)
        return self.log_softmax(out).unsqueeze(0)


form_model    = FormModel(d_form + 2 * d_content, d_form).to(device)
content_model = ContentModel(d_content + 2 * d_form, d_content).to(device)

pad_vector   = pad_vector.to(device)
start_vector = start_vector.to(device)

num_params = (sum(p.numel() for p in form_model.parameters()    if p.requires_grad) +
              sum(p.numel() for p in content_model.parameters() if p.requires_grad))
print(f"Num params: {num_params:,}")

criterion = nn.NLLLoss(reduction="sum")
optimizer  = torch.optim.Adam(
    itertools.chain(form_model.parameters(), content_model.parameters()), lr=1e-4
)

# ─────────────────────────────────────────────
# Helpers: pad sequences to equal length
# ─────────────────────────────────────────────

def pad_to_length(seq: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad a (L, d_form) tensor along dim-0 with pad_vector rows."""
    while seq.shape[0] < target_len:
        seq = torch.cat([seq, pad_vector], dim=0)
    return seq

# ─────────────────────────────────────────────
# Core: process one (row, pair) example
# ─────────────────────────────────────────────

def process_pair(row: dict, pair: dict, forced: bool) -> torch.Tensor:
    """
    Runs the full cyclic prediction pass for a single (row, pair) and returns
    the combined loss tensor (not yet reduced across the batch).
    """
    row_form    = torch.tensor(row["form_sequence"],  dtype=torch.float32, device=device)
    row_content = torch.tensor(row["content_tensor"], dtype=torch.float32, device=device)
    pair_form    = torch.tensor(pair["form_sequence"],  dtype=torch.float32, device=device)
    pair_content = torch.tensor(pair["content_tensor"], dtype=torch.float32, device=device)

    # ── 1. Content prediction: row → pair ───────────────────────────────────
    maxlen = max(row_form.shape[0], pair_form.shape[0])
    X = torch.cat([
        pad_to_length(row_form,  maxlen),
        pad_to_length(pair_form, maxlen),
        row_content.expand(maxlen, -1),
    ], dim=1)
    pred_content_1 = content_model(X)
    loss = criterion(pred_content_1, pair_content.unsqueeze(0).long())

    # ── 2. Form prediction: row → pair ──────────────────────────────────────
    maxlen = row_form.shape[0]
    if forced:
        exp_pair_content = pair_content.expand(maxlen, -1)
    else:
        exp_pair_content = torch.argmax(pred_content_1.squeeze(), dim=0).expand(maxlen, -1).float()

    X = torch.cat([row_form, row_content.expand(maxlen, -1), exp_pair_content], dim=1)
    y = pair_form[:-1]
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(y.shape[0], device=device)
    pred_form_1 = form_model(X, y, tgt_mask=tgt_mask, tgt_is_causal=True)
    actual_form_1 = torch.argmax(pair_form[1:], dim=1)
    loss = loss + criterion(pred_form_1, actual_form_1)

    # ── 3. Content prediction: pair → row ───────────────────────────────────
    if forced:
        seq_a = pair_form
    else:
        seq_a = torch.cat([start_vector,
                           torch.softmax(pred_form_1, dim=-1)], dim=0)
    maxlen = max(seq_a.shape[0], row_form.shape[0])
    X = torch.cat([
        pad_to_length(seq_a,    maxlen),
        pad_to_length(row_form, maxlen),
        pair_content.expand(maxlen, -1),
    ], dim=1)
    pred_content_2 = content_model(X)
    loss = loss + criterion(pred_content_2, row_content.unsqueeze(0).long())

    # ── 4. Form prediction: pair → row ──────────────────────────────────────
    maxlen = pair_form.shape[0]
    if forced:
        exp_row_content = row_content.expand(maxlen, -1)
    else:
        exp_row_content = torch.argmax(pred_content_2.squeeze(), dim=0).expand(maxlen, -1).float()

    X = torch.cat([pair_form, pair_content.expand(maxlen, -1), exp_row_content], dim=1)
    y = row_form[:-1]
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(y.shape[0], device=device)
    pred_form_2 = form_model(X, y, tgt_mask=tgt_mask, tgt_is_causal=True)
    actual_form_2 = torch.argmax(row_form[1:], dim=1)
    loss = loss + criterion(pred_form_2, actual_form_2)

    return loss

# ─────────────────────────────────────────────
# Train / eval loop using DataLoader
# ─────────────────────────────────────────────

def run_epoch(
    dataloader: DataLoader,
    forced: bool,
    training: bool,
) -> float:
    form_model.train(training)
    content_model.train(training)

    total_loss = 0.0
    total_pairs = 0
    pbar = tqdm(dataloader, desc="train" if training else "eval", leave=False)

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in pbar:
            batch_loss = torch.tensor(0.0, device=device)
            for row, pair in batch:
                batch_loss = batch_loss + process_pair(row, pair, forced=forced)

            if training:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            total_loss  += batch_loss.detach().item()
            total_pairs += len(batch)
            pbar.set_postfix(loss=f"{total_loss / total_pairs:.4f}")

    return total_loss / max(total_pairs, 1)

# ─────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────

def eval_inflection(row: dict) -> torch.Tensor:
    form_model.eval()
    root_form    = torch.tensor(get_form_sequence("*" + row["root"]).tolist(),
                                dtype=torch.float32, device=device)
    row_content  = torch.tensor(row["content_tensor"], dtype=torch.float32, device=device)
    root_content = get_content_tensor("ROOT").to(device)

    maxlen = root_form.shape[0]
    X = torch.cat([root_form,
                   root_content.expand(maxlen, -1),
                   row_content.expand(maxlen, -1)], dim=1)
    y = start_vector.clone()

    with torch.no_grad():
        for _ in range(100):
            pred = form_model(X, y)
            next_label = torch.argmax(pred[-1])
            next_vec = torch.zeros(1, d_form, device=device)
            next_vec[0, next_label] = 1.0
            y = torch.cat([y, next_vec], dim=0)
            if next_label == form_unigram_vocab[GLOSS_END]:
                break
    return y.cpu()


def eval_content(row: dict) -> torch.Tensor:
    content_model.eval()
    root_form   = torch.tensor(get_form_sequence("*" + row["root"]).tolist(),
                               dtype=torch.float32, device=device)
    row_form    = torch.tensor(row["form_sequence"],  dtype=torch.float32, device=device)
    root_content = get_content_tensor("ROOT").to(device)

    maxlen = max(root_form.shape[0], row_form.shape[0])
    X = torch.cat([
        pad_to_length(root_form, maxlen),
        pad_to_length(row_form,  maxlen),
        root_content.expand(maxlen, -1),
    ], dim=1)
    with torch.no_grad():
        return content_model(X).cpu()


def decode_form_pred(pred: torch.Tensor) -> str:
    inv = {i: c for c, i in form_unigram_vocab.items()}
    return "".join(inv[torch.argmax(tok).item()] for tok in pred[1:-1])


def decode_content_pred(pred: torch.Tensor) -> str:
    inv = sorted((i, c) for c, i in content_unigram_vocab.items())
    pred = pred.squeeze()
    return ";".join(c for is_present, (_, c) in zip(torch.argmax(pred, dim=0), inv) if is_present)


def get_accuracies(epoch: int, df_dict: dict[str, tuple]) -> dict[str, float]:
    accs = {}
    with torch.no_grad():
        for split_name, (split_df, split_dataset) in df_dict.items():
            rows = list(split_dataset)
            preds         = [decode_form_pred(eval_inflection(r))    for r in tqdm(rows, desc=f"{split_name} form")]
            pred_contents = [decode_content_pred(eval_content(r))    for r in tqdm(rows, desc=f"{split_name} content")]
            forms  = [r["form"]    for r in rows]
            contents = [r["content"] for r in rows]

            accuracy = sum(p == f for p, f in zip(preds, forms)) / len(forms)
            accs[split_name] = accuracy
            print(f"{split_name} accuracy: {accuracy:.2%}")

            out_df = pd.DataFrame({
                "form":         forms,
                "content":      contents,
                "pred":         preds,
                "pred_content": pred_contents,
            })
            out_df.to_csv(f"{split_name}_out.epoch{epoch}.tsv", sep="\t", index=False)

            for i, row in enumerate(out_df.sample(n=min(5, len(out_df)), random_state=42).itertuples()):
                writer.add_text(f"Example form {i} (g.t.|pred)/{split_name}",
                                f"{row.form} | {row.pred}", epoch)
                writer.add_text(f"Example content {i} (g.t.|pred)/{split_name}",
                                f"{row.content} | {row.pred_content}", epoch)
    return accs

# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────

train_loader = make_dataloader(train_dataset, train_root_index, shuffle=True)
dev_loader   = make_dataloader(dev_dataset,   dev_root_index,   shuffle=False)

# ── Forced pre-training ──────────────────────
print(f"\n=== Forced pre-training ({FORCED_PRETRAIN_EPOCHS} epoch(s)) ===")
for ep in range(FORCED_PRETRAIN_EPOCHS):
    loss = run_epoch(train_loader, forced=True, training=True)
    print(f"  Pre-train epoch {ep+1}: loss={loss:.4f}")

# Initial dev evaluation
print("\n=== Initial dev evaluation ===")
dev_loss_init = run_epoch(dev_loader, forced=True, training=False)
print(f"  Dev loss (forced): {dev_loss_init:.4f}")

# ── Main loop ────────────────────────────────
best_dev, best_dev_idx = float("inf"), -1

for cur_idx in itertools.count():
    if cur_idx - best_dev_idx > EARLY_STOP_THRESHOLD:
        print(f"Early stopping at iteration {cur_idx}.")
        break

    # Train
    train_loss = run_epoch(train_loader, forced=False, training=True)
    writer.add_scalar("Loss/train", train_loss, cur_idx)

    # Dev
    dev_loss = run_epoch(dev_loader, forced=False, training=False)
    writer.add_scalar("Loss/dev", dev_loss, cur_idx)

    print(f"[{cur_idx}] train_loss={train_loss:.4f}  dev_loss={dev_loss:.4f}")

    # Checkpoint
    if dev_loss < best_dev:
        best_dev, best_dev_idx = dev_loss, cur_idx
        print(f"  ✓ New best dev loss={best_dev:.4f} at iter {best_dev_idx}")
        iso = dt.now().isoformat().replace(":", "_").replace("-", "_")
        torch.save({
            "epoch":                    cur_idx,
            "dev_loss":                 best_dev,
            "date":                     dt.now().isoformat(),
            "form_model_state_dict":    form_model.state_dict(),
            "content_model_state_dict": content_model.state_dict(),
            "optimizer_state_dict":     optimizer.state_dict(),
        }, f"cyclic_inflection_models.{iso}.pt")

    # Periodic accuracy
    if (cur_idx + 1) % GET_ACCURACY_EPOCHS == 0:
        df_dict = {
            "train": (train_df, train_dataset),
            "dev":   (dev_df,   dev_dataset),
            "test":  (test_df,  test_dataset),
        }
        accs = get_accuracies(cur_idx, df_dict)
        for split, acc in accs.items():
            writer.add_scalar(f"Accuracy/{split}", acc, cur_idx)