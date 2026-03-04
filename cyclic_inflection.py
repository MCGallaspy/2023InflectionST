import itertools
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime as dt
from tqdm import tqdm

SAMPLE_RATE = 0.1
EARLY_STOP_THRESHOLD = 20
FORCED_PRETRAIN_EPOCHS = 50

train_df = pd.read_csv('part1/data/nav.trn', sep='\t', header=None, names=["root", "content", "form"])
train_df = train_df.sample(frac=SAMPLE_RATE)

data = []
for root in train_df.root.unique():
    data.append({
        "root": root,
        "content": "ROOT",
        "form": root,
    })
root_df = pd.DataFrame(data=data)
train_df = pd.concat([train_df, root_df], ignore_index=True)
print("Num train rows:", train_df.shape[0])

form_unigram_vocab = set()
for row in train_df.itertuples():
    form_unigram_vocab |= set(row.form)
GLOSS_START, GLOSS_END = "GLOSS_START", "GLOSS_END"
if len(form_unigram_vocab) % 2:
    form_unigram_vocab |= set((GLOSS_START, GLOSS_END, "OOV", "PAD", "FILL"))
else:
    form_unigram_vocab |= set((GLOSS_START, GLOSS_END, "OOV", "PAD"))

form_unigram_vocab = dict((w, i) for (i, w) in enumerate(form_unigram_vocab))
print("Form unigram vocab:", form_unigram_vocab)
print("Len form unigram vocab:", len(form_unigram_vocab))

content_unigram_vocab = set()
for row in train_df.itertuples():
    content_unigram_vocab |= set(row.content.split(";"))
if len(content_unigram_vocab) % 2:
    content_unigram_vocab |= set(("OOV",))
else:
    content_unigram_vocab |= set(("OOV", "FILL"))

content_unigram_vocab = dict((w, i) for (i, w) in enumerate(content_unigram_vocab))
print("Content unigram vocab:", content_unigram_vocab)
print("Len content unigram vocab:", len(content_unigram_vocab))

def get_form_sequence(word):
    idxs = []
    for c in word:
        try:
            idxs.append(form_unigram_vocab[c])
        except KeyError:
            idxs.append(form_unigram_vocab["OOV"])
    idxs = [form_unigram_vocab[GLOSS_START]] + idxs + [form_unigram_vocab[GLOSS_END]]
    return torch.stack([torch.Tensor([int(i == j) for i in range(len(form_unigram_vocab))]) for j in idxs])


def get_content_tensor(content):
    idxs = []
    contents = content.split(";")
    for c in contents:
        try:
            idxs.append(content_unigram_vocab[c])
        except KeyError:
            idxs.append(content_unigram_vocab["OOV"])
    idxs = set(idxs)
    return torch.Tensor([1 if i in idxs else 0 for i in range(len(content_unigram_vocab))])

    
train_df['content_tensor'] = train_df.content.apply(get_content_tensor)
train_df['form_sequence'] = train_df.form.apply(get_form_sequence)

test_df = pd.read_csv('part1/data/nav.tst', sep='\t', header=None, names=["root", "content", "form"])
test_df = test_df.sample(frac=SAMPLE_RATE)
print("Num test rows:", test_df.shape[0])
test_df['content_tensor'] = test_df.content.apply(get_content_tensor)
test_df['form_sequence'] = test_df.form.apply(get_form_sequence)

dev_df = pd.read_csv('part1/data/nav.dev', sep='\t', header=None, names=["root", "content", "form"])
dev_df = dev_df.sample(frac=SAMPLE_RATE)

data = []
for root in dev_df.root.unique():
    data.append({
        "root": root,
        "content": "ROOT",
        "form": root,
    })
root_df = pd.DataFrame(data=data)
dev_df = pd.concat([dev_df, root_df], ignore_index=True)
print("Num dev rows:", dev_df.shape[0])

dev_df['content_tensor'] = dev_df.content.apply(get_content_tensor)
dev_df['form_sequence'] = dev_df.form.apply(get_form_sequence)


class FormModel(torch.nn.Module):

    def __init__(self, d_input, num_glosses):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(
            in_channels=d_input,
            out_channels=256,
            kernel_size=3,
            padding='same',
        )
        self.conv2 = torch.nn.Conv1d(
            in_channels=256,
            out_channels=num_glosses,
            kernel_size=3,
            padding='same',
        )
        self.transformer = nn.Transformer(
            d_model=num_glosses,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            #dim_feedforward=64,
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)

        max_len = 50
        pos_encoding = torch.zeros(max_len, num_glosses, requires_grad=False)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, num_glosses, 2).float() * (-math.log(10000.0)) / num_glosses)
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, X, y, tgt_mask=None, tgt_is_causal=False):
        X = torch.permute(X.unsqueeze(0), dims=(0, 2, 1)) # To N, C, L
        X = self.conv1(X)
        X = self.conv2(X)
        X = torch.permute(X.squeeze(), dims=(1, 0)) # Back to L, C
        X += self.pos_encoding[:X.shape[0], :X.shape[1]]
        transformer_out = self.transformer(X, y, tgt_mask=tgt_mask, tgt_is_causal=tgt_is_causal)
        out = self.log_softmax(transformer_out)
        return out


d_form = len(form_unigram_vocab)
d_content = len(content_unigram_vocab)

form_model = FormModel(d_form + 2 * d_content, d_form)


class ContentModel(torch.nn.Module):

    def __init__(self, d_input, num_labels):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(
            in_channels=d_input,
            out_channels=256,
            kernel_size=3,
            padding='same',
        )
        self.conv2 = torch.nn.Conv1d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            padding='same',
        )
        self.conv3 = torch.nn.Conv1d(
            in_channels=256,
            out_channels=2,
            kernel_size=3,
            padding='same',
        )
        self.avg_pooling = nn.AdaptiveAvgPool1d(num_labels)
        self.log_softmax = nn.LogSoftmax(dim=0)

    def forward(self, X):
        X = torch.permute(X.unsqueeze(0), dims=(0, 2, 1)) # To N, C, L
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.conv3(X)
        X = self.avg_pooling(X)
        out = self.log_softmax(X)
        return out


content_model = ContentModel(d_content + 2 * d_form, d_content)

num_params = sum(p.numel() for p in form_model.parameters() if p.requires_grad)
num_params += sum(p.numel() for p in content_model.parameters() if p.requires_grad)
print(f"Num params={num_params}")

criterion = nn.NLLLoss(reduction='sum')
optimizer = torch.optim.Adam(itertools.chain(form_model.parameters(), content_model.parameters()), lr=1e-4)


def get_random_pair(row, source_df):
    root = row.root
    mask = source_df.root == root
    return source_df[mask].sample(n=1).iloc[0]


pad_vector = torch.Tensor([int(i == form_unigram_vocab['PAD']) for i in range(len(form_unigram_vocab))])
pad_vector = pad_vector.unsqueeze(0)

start_vector = torch.Tensor([int(i == form_unigram_vocab[GLOSS_START]) for i in range(len(form_unigram_vocab))])
start_vector = start_vector.unsqueeze(0)


def train_step(form_model, content_model, train_df, num_examples=1, num_epochs=1, forced=True):
    losses = []
    pbar = tqdm(total=num_examples * num_epochs)
    for n in range(num_epochs):
        epoch_loss = 0
        for row in train_df.sample(n=num_examples).itertuples():
            pair = get_random_pair(row, train_df)
            
            # First content prediction
            seqs = (row.form_sequence, pair.form_sequence)
            maxlen = max(s.shape[0] for s in seqs)
            vs = []
            for s in seqs:
                while s.shape[0] < maxlen:
                    s = torch.cat((s, pad_vector), dim=0)
                vs.append(s)
            X = torch.cat(vs, dim=1)
            expanded_content = row.content_tensor.expand(maxlen, -1)
            X = torch.cat([X, expanded_content], dim=1)
            pred_content = content_model(X)
            loss = criterion(pred_content, pair.content_tensor.unsqueeze(0).to(torch.long))

            # First form prediction
            maxlen = row.form_sequence.shape[0]
            if forced:
                expanded_row_content = row.content_tensor.expand(maxlen, -1)
                expanded_pair_content = pair.content_tensor.expand(maxlen, -1)
                X = torch.cat([row.form_sequence, expanded_row_content, expanded_pair_content], dim=1)
            else:
                expanded_row_content = row.content_tensor.expand(maxlen, -1)
                pred_content = torch.argmax(pred_content.squeeze(), dim=0)
                expanded_pair_content = pred_content.expand(maxlen, -1)
                X = torch.cat([row.form_sequence, expanded_row_content, expanded_pair_content], dim=1)
            y = pair.form_sequence[:-1, :]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(y.shape[0])
            pred_form = form_model(X, y, tgt_mask=tgt_mask, tgt_is_causal=True)
            actual_form = torch.argmax(pair.form_sequence[1:], axis=1)
            loss += criterion(pred_form, actual_form)
            
            # Second content prediction
            if forced:
                seqs = (pair.form_sequence, row.form_sequence)
            else:
                seqs = (torch.cat((start_vector, pred_form)), row.form_sequence)
            maxlen = max(s.shape[0] for s in seqs)
            vs = []
            for s in seqs:
                while s.shape[0] < maxlen:
                    s = torch.cat((s, pad_vector), dim=0)
                vs.append(s)
            X = torch.cat(vs, dim=1)
            expanded_content = pair.content_tensor.expand(maxlen, -1)
            X = torch.cat([X, expanded_content], dim=1)
            pred_content = content_model(X)
            loss += criterion(pred_content, row.content_tensor.unsqueeze(0).to(torch.long))
            
            # Second form prediction
            maxlen = pair.form_sequence.shape[0]
            if forced:
                expanded_row_content = row.content_tensor.expand(maxlen, -1)
                expanded_pair_content = pair.content_tensor.expand(maxlen, -1)
                X = torch.cat([pair.form_sequence, expanded_pair_content, expanded_row_content], dim=1)
            else:
                expanded_pair_content = pair.content_tensor.expand(maxlen, -1)
                pred_content = torch.argmax(pred_content.squeeze(), dim=0)
                expanded_row_content = pred_content.expand(maxlen, -1)
                X = torch.cat([pair.form_sequence, expanded_pair_content, expanded_row_content], dim=1)
            y = row.form_sequence[:-1, :]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(y.shape[0])
            pred_form = form_model(X, y, tgt_mask=tgt_mask, tgt_is_causal=True)
            actual_form = torch.argmax(row.form_sequence[1:], axis=1)
            loss += criterion(pred_form, actual_form)

            epoch_loss += loss.detach().item()
            if form_model.training:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            pbar.update(1)
        losses.append(epoch_loss / num_examples)
    return losses


def eval_inflection(form_model, row, eval_df):
    form_model.eval()
    root_form_sequence = get_form_sequence(row.root)
    maxlen = root_form_sequence.shape[0]
    expanded_row_content = row.content_tensor.expand(maxlen, -1)
    expanded_root_content = get_content_tensor("ROOT").expand(maxlen, -1)
    X = torch.cat([root_form_sequence, expanded_root_content, expanded_row_content], dim=1)
    y = torch.zeros((1, len(form_unigram_vocab)))
    y[0, form_unigram_vocab['GLOSS_START']] = 1
    while True:
        pred = form_model(X, y)
        next_y_label = torch.argmax(pred[-1, :])
        next_y = torch.zeros((1, y.shape[1]))
        next_y[0, next_y_label] = 1
        y = torch.concat([y, next_y], axis=0)
        if torch.argmax(y[-1, :len(form_unigram_vocab)]) == form_unigram_vocab[GLOSS_END]: break
        if y.shape[0] > 100: break
    return y


def decode_pred(pred):
     return ''.join([g
         for seq in pred[1:-1]
         for (g, i) in form_unigram_vocab.items()
         if i == torch.argmax(seq).item()])

train_losses = []
dev_losses = []

form_model.train()
content_model.train()
train_losses += train_step(
    form_model,
    content_model,
    train_df,
    num_examples=train_df.shape[0],
    num_epochs=FORCED_PRETRAIN_EPOCHS,
    forced=True,
)

print("Trainining losses:", train_losses)

form_model.eval()
content_model.eval()
with torch.no_grad():
    dev_losses += train_step(
        form_model,
        content_model,
        train_df,
        num_examples=dev_df.shape[0],
        num_epochs=1,
        forced=True,
    )

print("Dev losses:", dev_losses)


best_dev, best_dev_idx = 9e99, -1
cur_idx = 0

while True:
    if cur_idx - best_dev_idx > EARLY_STOP_THRESHOLD:
        if all(x >= best_dev for x in dev_losses[-EARLY_STOP_THRESHOLD:]):
            break

    form_model.train()
    content_model.train()
    train_losses += train_step(
        form_model,
        content_model,
        train_df,
        num_examples=train_df.shape[0],
        num_epochs=1,
        forced=False,
    )

    form_model.eval()
    content_model.eval()
    with torch.no_grad():
        dev_losses += train_step(
            form_model,
            content_model,
            dev_df,
            num_examples=dev_df.shape[0],
            num_epochs=1,
            forced=False,
        )

    if dev_losses[-1] < best_dev:
        best_dev = dev_losses[-1]
        best_dev_idx = cur_idx
        print(f"Best iteration: {best_dev_idx}, dev loss = {best_dev}")
        iso_time = dt.now().isoformat()
        iso_time_fp = iso_time.replace(":", "_")
        iso_time_fp = iso_time_fp.replace("-", "_")
        PATH = f"cyclic_inflection_models.{iso_time_fp}.pt"
        torch.save({
                    'epoch': cur_idx,
                    'dev_loss': best_dev,
                    'date': iso_time,
                    'form_model_state_dict': form_model.state_dict(),
                    'content_model_state_dict': form_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, PATH)
    cur_idx += 1

print("Training losses:", train_losses)
print("Dev losses:", dev_losses)


with torch.no_grad():
    dev_df['pred'] = dev_df.apply(lambda row: eval_inflection(form_model, row, dev_df), axis=1)
    dev_df.pred = dev_df.pred.apply(decode_pred)
    accuracy = np.sum(dev_df.pred == dev_df.form) / dev_df.shape[0]
    print(f"Dev accuracy: {accuracy:.2%}")
    dev_df.loc[:, ['form', 'pred']].to_csv("dev_out.tsv", sep="\t")

    test_df['pred'] = test_df.apply(lambda row: eval_inflection(form_model, row, test_df), axis=1)
    test_df.pred = test_df.pred.apply(decode_pred)
    accuracy = np.sum(test_df.pred == test_df.form) / test_df.shape[0]
    print(f"Test accuracy: {accuracy:.2%}")
    test_df.loc[:, ['form', 'pred']].to_csv("test_out.tsv", sep="\t")

    train_df['pred'] = train_df.apply(lambda row: eval_inflection(form_model, row, train_df), axis=1)
    train_df.pred = train_df.pred.apply(decode_pred)
    accuracy = np.sum(train_df.pred == train_df.form) / train_df.shape[0]
    print(f"Train accuracy: {accuracy:.2%}")
    train_df.loc[:, ['form', 'pred']].to_csv("train_out.tsv", sep="\t")

#iso_time = dt.now().isoformat()
#iso_time_fp = iso_time.replace(":", "_")
#iso_time_fp = iso_time_fp.replace("-", "_")
#PATH = f"cyclic_inflection_models.{iso_time_fp}.pt"
#torch.save({
#            'date': iso_time,
#            'form_model_state_dict': form_model.state_dict(),
#            'content_model_state_dict': form_model.state_dict(),
#            'optimizer_state_dict': optimizer.state_dict(),
#            }, PATH)