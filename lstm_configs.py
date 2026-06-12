"""
TFG - Testing 3+ LSTM configurations
Following Perello's suggestion (2026-04-14): try at least three more
configurations to check whether the results are roughly consistent.

Configurations tested:
  C1: 1-layer, hidden=32, dropout=0.2  (original baseline)
  C2: 2-layer, hidden=32, dropout=0.3
  C3: 1-layer, hidden=64, dropout=0.2
  C4: 2-layer, hidden=16, dropout=0.4  (more regularized, less capacity)

Metric: accuracy and cross-entropy on the test set, using the same
per-player split across all configurations.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)
np.random.seed(42)

df = pd.read_csv("/Users/racelabs/Desktop/TFG/data/rounds.csv")
df = df[df["decision"] != 0].copy()
df = df[~df["scenario"].str.startswith("4")].copy()
df = df.sort_values(["user", "game", "round"]).reset_index(drop=True)
df["dec_bin"] = (df["decision"] == 1).astype(int)
df["res_bin"] = (df["result"] == 1).astype(int)
df["market_bin"] = np.where(df["result"] == 1, df["dec_bin"], 1 - df["dec_bin"])

sequences = []
for game_id, group in df.groupby("game"):
    if len(group) < 10:
        continue
    sequences.append({
        "game": game_id,
        "user": group["user"].iloc[0],
        "decisions": group["dec_bin"].values,
        "results": group["res_bin"].values,
        "markets": group["market_bin"].values,
    })

users = sorted(set(s["user"] for s in sequences))
rng = np.random.default_rng(42)
rng.shuffle(users)
n_train = int(0.7 * len(users))
n_val = int(0.15 * len(users))
train_users = set(users[:n_train])
val_users = set(users[n_train:n_train + n_val])
test_users = set(users[n_train + n_val:])
train_seqs = [s for s in sequences if s["user"] in train_users]
val_seqs = [s for s in sequences if s["user"] in val_users]
test_seqs = [s for s in sequences if s["user"] in test_users]


class DecisionDataset(Dataset):
    def __init__(self, sequences, max_len=25):
        self.inputs = []
        self.targets = []
        self.lengths = []
        for s in sequences:
            n = len(s["decisions"])
            for t in range(1, n):
                start = max(0, t - max_len)
                inp = np.stack([
                    s["decisions"][start:t],
                    s["results"][start:t],
                    s["markets"][start:t]
                ], axis=1).astype(np.float32)
                self.inputs.append(inp)
                self.targets.append(s["decisions"][t])
                self.lengths.append(min(t, max_len))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.lengths[idx]


def collate_fn(batch):
    inputs, targets, lengths = zip(*batch)
    max_len = max(len(inp) for inp in inputs)
    padded = np.zeros((len(inputs), max_len, 3), dtype=np.float32)
    for i, inp in enumerate(inputs):
        padded[i, :len(inp)] = inp
    return (torch.tensor(padded),
            torch.tensor(targets, dtype=torch.float32),
            torch.tensor(lengths, dtype=torch.long))


class LSTMPredictor(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.fc(self.dropout(h[-1])).squeeze(-1)


def train_and_eval(hidden, num_layers, dropout, seed=42, n_epochs=25):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_ds = DecisionDataset(train_seqs)
    val_ds = DecisionDataset(val_seqs)
    test_ds = DecisionDataset(test_seqs)
    tr = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    va = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    te = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    model = LSTMPredictor(3, hidden, num_layers, dropout)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = float("inf")
    best_state = None
    for epoch in range(n_epochs):
        model.train()
        for x, y, l in tr:
            opt.zero_grad()
            loss = crit(model(x, l), y)
            loss.backward()
            opt.step()
        model.eval()
        v_loss = 0
        n_v = 0
        with torch.no_grad():
            for x, y, l in va:
                v_loss += crit(model(x, l), y).item() * len(y)
                n_v += len(y)
        v_loss /= n_v
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    correct = 0
    total = 0
    ce = 0
    with torch.no_grad():
        for x, y, l in te:
            logits = model(x, l)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += len(y)
            ce += crit(logits, y).item() * len(y)
    return correct / total, ce / total, best_val


configs = [
    ("C1: 1L h=32 drop=0.2", 32, 1, 0.2),
    ("C2: 2L h=32 drop=0.3", 32, 2, 0.3),
    ("C3: 1L h=64 drop=0.2", 64, 1, 0.2),
    ("C4: 2L h=16 drop=0.4", 16, 2, 0.4),
]

print(f"Train users {len(train_users)}, val {len(val_users)}, test {len(test_users)}")
print("\n=== 3+ LSTM CONFIGURATIONS ===")
results = []
for name, h, l, d in configs:
    acc, ce, val_loss = train_and_eval(h, l, d)
    print(f"{name:30s}  acc={acc:.4f}  ce={ce:.4f}  best_val_loss={val_loss:.4f}")
    results.append({"config": name, "hidden": h, "layers": l, "dropout": d,
                    "acc": acc, "ce": ce, "best_val_loss": val_loss})

pd.DataFrame(results).to_csv("/Users/racelabs/Desktop/TFG/data/lstm_configs_results.csv", index=False)
print("\nSaved: data/lstm_configs_results.csv")

accs = [r["acc"] for r in results]
print(f"\nAccuracy range: [{min(accs):.4f}, {max(accs):.4f}]  spread={max(accs)-min(accs):.4f}")
print("If spread < 0.01, the results are consistent.")
