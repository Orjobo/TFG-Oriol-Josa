"""
TFG - LSTM for predicting human decisions.
Benchmarked against the random, MI and WSLS baselines.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- Load data ---
df = pd.read_csv("/Users/racelabs/Desktop/TFG/data/rounds.csv")
df = df[df["decision"] != 0].copy()
df = df[~df["scenario"].str.startswith("4")].copy()
df = df.sort_values(["user", "game", "round"]).reset_index(drop=True)
df["dec_bin"] = (df["decision"] == 1).astype(int)
df["res_bin"] = (df["result"] == 1).astype(int)
df["market_bin"] = np.where(df["result"] == 1, df["dec_bin"], 1 - df["dec_bin"])

print(f"Decisions: {len(df)}, Users: {df['user'].nunique()}")

# --- Build per-game sequences (25 rounds each) ---
sequences = []
for game_id, group in df.groupby("game"):
    if len(group) < 10:
        continue
    decs = group["dec_bin"].values
    results = group["res_bin"].values
    markets = group["market_bin"].values
    user = group["user"].iloc[0]
    sequences.append({
        "game": game_id,
        "user": user,
        "decisions": decs,
        "results": results,
        "markets": markets
    })

print(f"Sequences (games): {len(sequences)}")

# --- Split by user, not by game, so the same player never leaks across sets ---
users = list(set(s["user"] for s in sequences))
np.random.seed(42)
np.random.shuffle(users)
n_train = int(0.7 * len(users))
n_val = int(0.15 * len(users))
train_users = set(users[:n_train])
val_users = set(users[n_train:n_train+n_val])
test_users = set(users[n_train+n_val:])

train_seqs = [s for s in sequences if s["user"] in train_users]
val_seqs = [s for s in sequences if s["user"] in val_users]
test_seqs = [s for s in sequences if s["user"] in test_users]

print(f"Train: {len(train_seqs)} games ({len(train_users)} users)")
print(f"Val: {len(val_seqs)} games ({len(val_users)} users)")
print(f"Test: {len(test_seqs)} games ({len(test_users)} users)")

# --- Baselines ---
def baseline_random(sequences):
    """Predict using the global p(up)."""
    all_decs = np.concatenate([s["decisions"] for s in sequences])
    p_up = all_decs.mean()
    correct = 0
    total = 0
    ce_sum = 0
    for s in sequences:
        for t in range(1, len(s["decisions"])):
            target = s["decisions"][t]
            pred = p_up
            correct += (1 if (pred > 0.5) == target else 0)
            ce_sum += -(target * np.log(pred + 1e-8) + (1-target) * np.log(1-pred + 1e-8))
            total += 1
    return correct / total, ce_sum / total

def baseline_mi(sequences):
    """Predict D_n = M_{n-1}."""
    correct = 0
    total = 0
    ce_sum = 0
    for s in sequences:
        for t in range(1, len(s["decisions"])):
            target = s["decisions"][t]
            pred_class = s["markets"][t-1]
            pred = 0.714 if pred_class == 1 else (1 - 0.531)  # probabilities from the paper
            correct += (1 if pred_class == target else 0)
            ce_sum += -(target * np.log(pred + 1e-8) + (1-target) * np.log(1-pred + 1e-8))
            total += 1
    return correct / total, ce_sum / total

def baseline_wsls(sequences):
    """Repeat on a win, switch on a loss."""
    correct = 0
    total = 0
    ce_sum = 0
    for s in sequences:
        for t in range(1, len(s["decisions"])):
            target = s["decisions"][t]
            if s["results"][t-1] == 1:
                pred_class = s["decisions"][t-1]  # repeat
                pred = 0.682
            else:
                pred_class = 1 - s["decisions"][t-1]  # switch
                pred = 0.579
            # Flip pred to match the predicted class
            if pred_class == 0:
                pred = 1 - pred
            correct += (1 if pred_class == target else 0)
            ce_sum += -(target * np.log(pred + 1e-8) + (1-target) * np.log(1-pred + 1e-8))
            total += 1
    return correct / total, ce_sum / total

print("\n--- Baselines (test set) ---")
acc_rand, ce_rand = baseline_random(test_seqs)
acc_mi, ce_mi = baseline_mi(test_seqs)
acc_wsls, ce_wsls = baseline_wsls(test_seqs)
print(f"Random:    acc={acc_rand:.4f}, cross-entropy={ce_rand:.4f}")
print(f"MI:        acc={acc_mi:.4f}, cross-entropy={ce_mi:.4f}")
print(f"WSLS:      acc={acc_wsls:.4f}, cross-entropy={ce_wsls:.4f}")

# --- PyTorch dataset ---
class DecisionDataset(Dataset):
    def __init__(self, sequences, max_len=25):
        self.inputs = []
        self.targets = []
        self.lengths = []
        for s in sequences:
            n = len(s["decisions"])
            # Input: (decision, result, market) at each timestep
            # Target: the decision at the next timestep
            for t in range(1, n):
                seq_len = min(t, max_len)
                start = max(0, t - max_len)
                inp = np.stack([
                    s["decisions"][start:t],
                    s["results"][start:t],
                    s["markets"][start:t]
                ], axis=1).astype(np.float32)
                self.inputs.append(inp)
                self.targets.append(s["decisions"][t])
                self.lengths.append(seq_len)

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

# --- Model LSTM ---
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        # Pack padded sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, (hidden, _) = self.lstm(packed)
        # Use the last hidden state
        out = self.dropout(hidden[-1])
        return self.fc(out).squeeze(-1)

# --- Train ---
print("\nPreparing datasets...")
train_ds = DecisionDataset(train_seqs)
val_ds = DecisionDataset(val_seqs)
test_ds = DecisionDataset(test_seqs)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

model = LSTMPredictor(input_size=3, hidden_size=32, num_layers=1, dropout=0.2)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

N_EPOCHS = 30
train_losses = []
val_losses = []

print(f"\nTraining LSTM ({N_EPOCHS} epochs)...")
for epoch in range(N_EPOCHS):
    # Train
    model.train()
    epoch_loss = 0
    n_batches = 0
    for inputs, targets, lengths in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs, lengths)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    train_losses.append(epoch_loss / n_batches)

    # Val
    model.eval()
    val_loss = 0
    n_val = 0
    with torch.no_grad():
        for inputs, targets, lengths in val_loader:
            outputs = model(inputs, lengths)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            n_val += 1
    val_losses.append(val_loss / n_val)

    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}: train_loss={train_losses[-1]:.4f}, val_loss={val_losses[-1]:.4f}")

# --- Evaluate on the test set ---
model.eval()
all_preds = []
all_targets = []
test_loss = 0
n_test = 0
with torch.no_grad():
    for inputs, targets, lengths in test_loader:
        outputs = model(inputs, lengths)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        n_test += 1
        probs = torch.sigmoid(outputs)
        all_preds.extend(probs.numpy())
        all_targets.extend(targets.numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

acc_lstm = ((all_preds > 0.5) == all_targets).mean()
ce_lstm = test_loss / n_test

# Cross-entropy computed by hand so it lines up with the baselines
ce_lstm_manual = -np.mean(
    all_targets * np.log(all_preds + 1e-8) +
    (1 - all_targets) * np.log(1 - all_preds + 1e-8)
)

print(f"\n{'='*60}")
print("FINAL RESULTS")
print("="*60)
print(f"\n{'Model':<15} {'Accuracy':>10} {'Cross-Entropy':>15}")
print("-"*42)
print(f"{'Random':<15} {acc_rand:>10.4f} {ce_rand:>15.4f}")
print(f"{'MI':<15} {acc_mi:>10.4f} {ce_mi:>15.4f}")
print(f"{'WSLS':<15} {acc_wsls:>10.4f} {ce_wsls:>15.4f}")
print(f"{'LSTM':<15} {acc_lstm:>10.4f} {ce_lstm_manual:>15.4f}")

# --- Figures ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("LSTM vs Baselines — Prediccio de decisions", fontsize=14, fontweight="bold")

# Panel 1: training curves
ax = axes[0]
ax.plot(train_losses, label="Train", color="steelblue")
ax.plot(val_losses, label="Validation", color="coral")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (BCE)")
ax.set_title("Corbes d'entrenament")
ax.legend()

# Panel 2: accuracy comparison
ax = axes[1]
models = ["Aleatori", "MI", "WSLS", "LSTM"]
accs = [acc_rand, acc_mi, acc_wsls, acc_lstm]
colors = ["gray", "steelblue", "coral", "mediumpurple"]
bars = ax.bar(models, accs, color=colors, edgecolor="white", alpha=0.8)
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy: LSTM vs Baselines")
ax.axhline(0.5, color="gray", ls="--", alpha=0.3)
for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10)

# Panel 3: cross-entropy comparison
ax = axes[2]
ces = [ce_rand, ce_mi, ce_wsls, ce_lstm_manual]
bars = ax.bar(models, ces, color=colors, edgecolor="white", alpha=0.8)
ax.set_ylabel("Cross-Entropy")
ax.set_title("Cross-Entropy: LSTM vs Baselines")
for bar, val in zip(bars, ces):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig("/Users/racelabs/Desktop/TFG/lstm_results.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure: /Users/racelabs/Desktop/TFG/lstm_results.png")
print("DONE")
