"""
Этап 5 (бонус, +5 баллов). Ансамбль RNN + CNN с весами, подобранными на валидации.

Идея ансамбля:
    - RNN (здесь: двунаправленный LSTM) хорошо улавливает порядок слов
      и контекст — для фраз вроде "не поверите, что случилось" важно,
      что слова идут именно в таком порядке.
    - CNN (свёрточная сеть с несколькими размерами окна — 3, 4, 5 слов)
      хорошо находит характерные локальные словосочетания-маркеры
      ("скидка %", "переходи по ссылке", "количество мест") независимо
      от их позиции в предложении.
    - У RNN и CNN разные "слепые зоны", поэтому усреднение их вероятностей
      с весом, подобранным на валидационной выборке (а не на глаз),
      может дать точность выше, чем у каждой модели по отдельности.

Обе модели учат embedding-слой с нуля на нашем train-датасете (это НЕ
трансформер и не предобученная языковая модель — ровно то, что разрешено
заданием).

Запуск:
    python src/train_rnn_cnn_ensemble.py

Результат:
    - метрики RNN, CNN и ансамбля в консоли
    - results/model_comparison.png — сравнение accuracy всех 4 моделей
    - results/ensemble_confusion_matrix.png
    - results/ensemble_misclassified.csv — ошибки ансамбля для анализа
    - results/ensemble_metrics.json
"""

import json
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

from train_baseline import (
    plot_confusion_matrix, CLASS_COLORS, INK, INK_SECONDARY, INK_MUTED, GRID, SURFACE,
)

torch.manual_seed(42)
np.random.seed(42)

MAX_LEN = 30
EMBED_DIM = 100
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_COLORS = {
    "Baseline": "#2a78d6",
    "RNN": "#eb6834",
    "CNN": "#1baf7a",
    "Ансамбль": "#eda100",
}

# ---------------------------------------------------------------------------
# 1. Данные и словарь
# ---------------------------------------------------------------------------

train_df = pd.read_csv("data/train.csv").dropna(subset=["clean_text"])
val_df = pd.read_csv("data/val.csv").dropna(subset=["clean_text"])
test_df = pd.read_csv("data/test.csv").dropna(subset=["clean_text"])

le = LabelEncoder()
y_train = le.fit_transform(train_df["label"])  # alphabetically: clickbait=0, spam=1
y_val = le.transform(val_df["label"])
y_test = le.transform(test_df["label"])
print(f"Классы: {dict(zip(le.classes_, range(len(le.classes_))))}")

counter = Counter()
for text in train_df["clean_text"]:
    counter.update(text.split())

MIN_FREQ = 2
vocab = ["<PAD>", "<UNK>"] + [w for w, c in counter.items() if c >= MIN_FREQ]
word2id = {w: i for i, w in enumerate(vocab)}
VOCAB_SIZE = len(vocab)
print(f"Размер словаря (min_freq={MIN_FREQ}): {VOCAB_SIZE}")


def encode(text: str) -> list[int]:
    ids = [word2id.get(w, 1) for w in text.split()[:MAX_LEN]]
    ids += [0] * (MAX_LEN - len(ids))
    return ids


def to_tensor(df):
    X = torch.tensor([encode(t) for t in df["clean_text"]], dtype=torch.long)
    return X


X_train = to_tensor(train_df)
X_val = to_tensor(val_df)
X_test = to_tensor(test_df)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_val_t = torch.tensor(y_val, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val_t), batch_size=BATCH_SIZE)
test_loader = DataLoader(TensorDataset(X_test, y_test_t), batch_size=BATCH_SIZE)


# ---------------------------------------------------------------------------
# 2. Модели
# ---------------------------------------------------------------------------

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim=64, n_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        # h_n: (num_directions, batch, hidden) -> конкатенируем forward/backward
        h = torch.cat([h_n[0], h_n[1]], dim=1)
        h = self.dropout(h)
        return self.fc(h)


class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_filters=100, kernel_sizes=(3, 4, 5), n_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k, padding=k // 2) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), n_classes)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)  # (batch, embed_dim, seq_len)
        pooled = [torch.relu(conv(emb)).max(dim=2).values for conv in self.convs]
        h = torch.cat(pooled, dim=1)
        h = self.dropout(h)
        return self.fc(h)


# ---------------------------------------------------------------------------
# 3. Обучение
# ---------------------------------------------------------------------------

class LiveTrainingPlot:
    """
    Всплывающее окно matplotlib, которое обновляется ПОСЛЕ КАЖДОЙ ЭПОХИ —
    так процесс обучения видно вживую: как падает loss и растёт accuracy
    на валидации у RNN и CNN, эпоха за эпохой, на одном графике для сравнения.
    """

    def __init__(self):
        plt.ion()
        self.fig, (self.ax_loss, self.ax_acc) = plt.subplots(
            1, 2, figsize=(11, 4.5), facecolor=SURFACE
        )
        for ax, title in ((self.ax_loss, "Train loss по эпохам"), (self.ax_acc, "Val accuracy по эпохам")):
            ax.set_facecolor(SURFACE)
            ax.set_title(title, color=INK, fontsize=11)
            ax.set_xlabel("Эпоха", color=INK_SECONDARY)
            ax.grid(color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(length=0, colors=INK_SECONDARY)
        self.ax_acc.set_ylim(0, 1.05)
        self.fig.suptitle("Процесс обучения RNN и CNN (эпоха за эпохой)", color=INK, fontsize=13)

        self.history = {}
        self.lines_loss = {}
        self.lines_acc = {}
        try:
            self.fig.canvas.manager.set_window_title("Обучение моделей")
        except Exception:
            pass
        try:
            plt.show(block=False)
        except Exception:
            pass

    def update(self, name, epoch, loss, val_acc):
        h = self.history.setdefault(name, {"epoch": [], "loss": [], "val_acc": []})
        h["epoch"].append(epoch)
        h["loss"].append(loss)
        h["val_acc"].append(val_acc)
        color = MODEL_COLORS.get(name, INK_MUTED)

        if name not in self.lines_loss:
            (self.lines_loss[name],) = self.ax_loss.plot(
                h["epoch"], h["loss"], color=color, linewidth=2, marker="o", markersize=4, label=name
            )
            (self.lines_acc[name],) = self.ax_acc.plot(
                h["epoch"], h["val_acc"], color=color, linewidth=2, marker="o", markersize=4, label=name
            )
            self.ax_loss.legend(frameon=False, labelcolor=INK_SECONDARY)
            self.ax_acc.legend(frameon=False, labelcolor=INK_SECONDARY)
        else:
            self.lines_loss[name].set_data(h["epoch"], h["loss"])
            self.lines_acc[name].set_data(h["epoch"], h["val_acc"])

        self.ax_loss.relim()
        self.ax_loss.autoscale_view()
        self.ax_acc.relim()
        self.ax_acc.autoscale_view()

        self.fig.tight_layout()
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)
        except Exception:
            pass

    def save(self, path):
        self.fig.savefig(path, dpi=150, facecolor=SURFACE)


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total


def train_model(model, name, live_plot=None):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader)
        print(f"[{name}] epoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")
        if live_plot is not None:
            live_plot.update(name, epoch, avg_loss, val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def get_probs(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(X.to(DEVICE))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


live_plot = LiveTrainingPlot()

print("\n--- Обучение RNN ---")
rnn_model = RNNClassifier(VOCAB_SIZE, EMBED_DIM)
rnn_model = train_model(rnn_model, "RNN", live_plot)

print("\n--- Обучение CNN ---")
cnn_model = CNNClassifier(VOCAB_SIZE, EMBED_DIM)
cnn_model = train_model(cnn_model, "CNN", live_plot)

live_plot.save("results/training_curves.png")
print("\nГрафик обучения сохранён: results/training_curves.png")

# ---------------------------------------------------------------------------
# 4. Подбор веса ансамбля на валидации
# ---------------------------------------------------------------------------

rnn_val_probs = get_probs(rnn_model, X_val)
cnn_val_probs = get_probs(cnn_model, X_val)
rnn_test_probs = get_probs(rnn_model, X_test)
cnn_test_probs = get_probs(cnn_model, X_test)

best_w, best_acc = 0.5, 0.0
for w in np.arange(0.0, 1.01, 0.05):
    mix = w * rnn_val_probs + (1 - w) * cnn_val_probs
    acc = accuracy_score(y_val, mix.argmax(axis=1))
    if acc > best_acc:
        best_acc, best_w = acc, w

print(f"\nЛучший вес на валидации: w_rnn={best_w:.2f}, w_cnn={1 - best_w:.2f} (val_acc={best_acc:.4f})")

rnn_test_acc = accuracy_score(y_test, rnn_test_probs.argmax(axis=1))
cnn_test_acc = accuracy_score(y_test, cnn_test_probs.argmax(axis=1))
ensemble_test_probs = best_w * rnn_test_probs + (1 - best_w) * cnn_test_probs
ensemble_test_pred = ensemble_test_probs.argmax(axis=1)
ensemble_test_acc = accuracy_score(y_test, ensemble_test_pred)

with open("results/baseline_metrics.json", encoding="utf-8") as f:
    baseline_acc = json.load(f)["test_accuracy"]

print(f"\nAccuracy на тесте:")
print(f"  Baseline (TF-IDF+LogReg): {baseline_acc:.4f}")
print(f"  RNN:                      {rnn_test_acc:.4f}")
print(f"  CNN:                      {cnn_test_acc:.4f}")
print(f"  Ансамбль (RNN+CNN):       {ensemble_test_acc:.4f}")

print("\nClassification report ансамбля (test):")
print(classification_report(y_test, ensemble_test_pred, target_names=le.classes_))

# ---------------------------------------------------------------------------
# 5. Графики и сохранение результатов
# ---------------------------------------------------------------------------

labels = list(le.classes_)
cm = confusion_matrix(y_test, ensemble_test_pred)
plot_confusion_matrix(
    cm, labels, "results/ensemble_confusion_matrix.png",
    "Ансамбль RNN+CNN: confusion matrix",
)

fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
names = ["Baseline", "RNN", "CNN", "Ансамбль"]
values = [baseline_acc, rnn_test_acc, cnn_test_acc, ensemble_test_acc]
colors = [MODEL_COLORS[n] for n in names]
bars = ax.bar(names, values, color=colors, width=0.55)
ax.bar_label(bars, fmt="%.3f", padding=3, color=INK_SECONDARY, fontsize=10)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Accuracy на тесте", color=INK_SECONDARY)
ax.set_title("Сравнение моделей классификации", color=INK, fontsize=12, pad=12)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0, colors=INK_SECONDARY)
fig.tight_layout()
fig.savefig("results/model_comparison.png", dpi=150, facecolor=SURFACE)
plt.close(fig)

test_df = test_df.copy()
test_df["predicted"] = le.inverse_transform(ensemble_test_pred)
mistakes = test_df[test_df["label"] != test_df["predicted"]]
mistakes[["text", "clean_text", "label", "predicted", "is_noisy", "is_label_flipped"]].to_csv(
    "results/ensemble_misclassified.csv", index=False, encoding="utf-8"
)
print(f"\nОшибок ансамбля на тесте: {len(mistakes)} из {len(test_df)}")

with open("results/ensemble_metrics.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "baseline_test_accuracy": baseline_acc,
            "rnn_test_accuracy": rnn_test_acc,
            "cnn_test_accuracy": cnn_test_acc,
            "ensemble_test_accuracy": ensemble_test_acc,
            "best_weight_rnn": float(best_w),
            "best_weight_cnn": float(1 - best_w),
            "vocab_size": VOCAB_SIZE,
            "n_errors_ensemble": int(len(mistakes)),
        },
        f,
        ensure_ascii=False,
        indent=2,
    )
print("Сохранено: results/model_comparison.png, results/ensemble_confusion_matrix.png,")
print("           results/ensemble_misclassified.csv, results/ensemble_metrics.json")

print("\nОкно с графиком обучения осталось открытым — закройте его, чтобы завершить скрипт.")
try:
    plt.ioff()
    plt.show(block=True)
except Exception:
    pass
