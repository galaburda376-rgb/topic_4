"""
Этап 4. Базовая классификация текста (метод без трансформеров).

Метод: TF-IDF + логистическая регрессия.

Почему именно этот метод (для отчёта):
    - TF-IDF (Term Frequency - Inverse Document Frequency) превращает текст
      в вектор чисел: вес слова тем выше, чем чаще оно встречается в данном
      сообщении и чем реже — во всех остальных. Для спама и кликбейта это
      работает хорошо, потому что у каждого класса есть характерные "маркерные"
      слова ("скидка", "доход", "переходи" для спама; "шок", "не поверите",
      "фактов" для кликбейта), которые TF-IDF как раз и выделяет высоким весом.
    - Логистическая регрессия поверх TF-IDF — стандартный сильный baseline
      для текстовой классификации на коротких сообщениях: мало параметров,
      быстро обучается, не переобучается на датасете из нескольких тысяч
      примеров (в отличие от нейросетей, которым нужно больше данных),
      и даёт вероятности классов (понадобятся дальше для ансамбля).
    - Это ровно один из методов, которые рассматривались на вебинаре 4,
      и явно не трансформер.

Запуск:
    python src/train_baseline.py

Результат:
    - метрики в консоли (accuracy, precision, recall, f1, confusion matrix)
    - results/baseline_misclassified.csv — ошибки модели для анализа
    - results/baseline_metrics.json — метрики для итогового отчёта
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Единая цветовая схема для всего проекта: spam = синий, clickbait = оранжевый.
# Используется на всех графиках (baseline, нейросети, ансамбль), чтобы цвет
# класса был одинаковым везде.
COLOR_SPAM = "#2a78d6"
COLOR_CLICKBAIT = "#eb6834"
CLASS_COLORS = {"spam": COLOR_SPAM, "clickbait": COLOR_CLICKBAIT}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
BLUE_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_seq", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#104281"]
)


def plot_confusion_matrix(cm, labels, out_path, title):
    fig, ax = plt.subplots(figsize=(5, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(cm, cmap=BLUE_SEQUENTIAL)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, color=INK_SECONDARY)
    ax.set_yticklabels(labels, color=INK_SECONDARY)
    ax.set_xlabel("Предсказано", color=INK_SECONDARY)
    ax.set_ylabel("Истинный класс", color=INK_SECONDARY)
    ax.set_title(title, color=INK, fontsize=12, pad=12)

    vmax = cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            text_color = SURFACE if value > vmax * 0.5 else INK
            ax.text(j, i, str(value), ha="center", va="center",
                     color=text_color, fontsize=13, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_per_class_metrics(report_dict, labels, out_path, title):
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(metrics))
    width = 0.32

    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for i, label in enumerate(labels):
        values = [report_dict[label][m] for m in metrics]
        offset = (i - (len(labels) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=label,
                       color=CLASS_COLORS.get(label, INK_MUTED))
        ax.bar_label(bars, fmt="%.2f", padding=3, color=INK_SECONDARY, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1-score"], color=INK_SECONDARY)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Значение", color=INK_SECONDARY)
    ax.set_title(title, color=INK, fontsize=12, pad=12)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY)

    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, color=INK_MUTED)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)

def main():
    train_df = pd.read_csv("data/train.csv").dropna(subset=["clean_text"])
    val_df = pd.read_csv("data/val.csv").dropna(subset=["clean_text"])
    test_df = pd.read_csv("data/test.csv").dropna(subset=["clean_text"])

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_train = vectorizer.fit_transform(train_df["clean_text"])
    X_val = vectorizer.transform(val_df["clean_text"])
    X_test = vectorizer.transform(test_df["clean_text"])

    y_train = train_df["label"]
    y_val = val_df["label"]
    y_test = test_df["label"]

    model = LogisticRegression(max_iter=1000, C=5.0)
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"Accuracy на валидации: {val_acc:.4f}")
    print(f"Accuracy на тесте:     {test_acc:.4f}\n")
    print("Classification report (test):")
    print(classification_report(y_test, test_pred))
    print("Confusion matrix (test), строки=истина, столбцы=предсказание:")
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, test_pred, labels=labels)
    print(f"labels: {labels}")
    print(cm)

    report_dict = classification_report(y_test, test_pred, output_dict=True)
    plot_confusion_matrix(
        cm, labels, "results/baseline_confusion_matrix.png",
        "Baseline (TF-IDF + LogReg): confusion matrix",
    )
    plot_per_class_metrics(
        report_dict, labels, "results/baseline_metrics.png",
        "Baseline (TF-IDF + LogReg): метрики по классам",
    )
    print("Графики сохранены: results/baseline_confusion_matrix.png, results/baseline_metrics.png")

    # Сохраняем ошибки классификации для этапа анализа ошибок
    test_df = test_df.copy()
    test_df["predicted"] = test_pred
    mistakes = test_df[test_df["label"] != test_df["predicted"]]
    mistakes[["text", "clean_text", "label", "predicted"]].to_csv(
        "results/baseline_misclassified.csv", index=False, encoding="utf-8"
    )
    print(f"\nОшибок на тесте: {len(mistakes)} из {len(test_df)}")

    with open("results/baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "TF-IDF + LogisticRegression",
                "val_accuracy": val_acc,
                "test_accuracy": test_acc,
                "n_errors": int(len(mistakes)),
                "n_test": int(len(test_df)),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("Сохранено: results/baseline_misclassified.csv, results/baseline_metrics.json")


if __name__ == "__main__":
    main()
