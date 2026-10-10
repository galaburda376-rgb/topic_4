"""
Этап 6. Анализ неправильно классифицированных объектов.

Запуск:
    python src/error_analysis.py

Что делает:
    - загружает ошибки ансамбля (results/ensemble_misclassified.csv)
    - делит их на две группы:
        1) ошибки из-за шума РАЗМЕТКИ (is_label_flipped=1) — модель права
           по смыслу текста, а "эталонная" метка в датасете намеренно неверна
        2) ошибки из-за реальной сложности текста (is_label_flipped=0) —
           модель действительно перепутала класс по смыслу
    - печатает несколько примеров каждой группы
    - сохраняет график состава ошибок results/error_composition.png
"""

import pandas as pd
import matplotlib.pyplot as plt

from train_baseline import SURFACE, INK, INK_SECONDARY, GRID

df = pd.read_csv("results/ensemble_misclassified.csv")
n_total = len(df)
n_label_noise = int(df["is_label_flipped"].sum())
n_real_errors = n_total - n_label_noise
n_text_noise_among_errors = int(df["is_noisy"].sum())

print(f"Всего ошибок ансамбля на тесте: {n_total}")
print(f"  из них из-за шума разметки (намеренно перепутанная метка): {n_label_noise} "
      f"({n_label_noise / n_total:.0%})")
print(f"  из них содержательные ошибки модели: {n_real_errors} ({n_real_errors / n_total:.0%})")
print(f"  из них дополнительно с опечатками в тексте: {n_text_noise_among_errors}")

print("\n--- Примеры ошибок из-за шума разметки (модель права по смыслу) ---")
label_noise_examples = df[df["is_label_flipped"] == 1].head(5)
for _, row in label_noise_examples.iterrows():
    print(f'  Текст:     "{row["text"]}"')
    print(f'  Метка в датасете (ошибочная): {row["label"]}  ->  Предсказание модели: {row["predicted"]}\n')

if n_real_errors > 0:
    print("--- Примеры содержательных ошибок модели ---")
    real_examples = df[df["is_label_flipped"] == 0].head(5)
    for _, row in real_examples.iterrows():
        print(f'  Текст:     "{row["text"]}"')
        print(f'  Истинная метка: {row["label"]}  ->  Предсказание модели: {row["predicted"]}\n')
else:
    print("\nСодержательных ошибок (не связанных с шумом разметки) на тесте не найдено:")
    print("модель верно разделяет spam и clickbait по смыслу текста во всех случаях,")
    print("кроме тех, где сама эталонная метка в датасете была намеренно испорчена.")

# ---------------------------------------------------------------------------
# График состава ошибок
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
categories = ["Шум разметки\n(метка неверна)", "Содержательная\nошибка модели"]
values = [n_label_noise, n_real_errors]
colors = ["#898781", "#e34948"]
bars = ax.bar(categories, values, color=colors, width=0.5)
ax.bar_label(bars, padding=3, color=INK_SECONDARY, fontsize=11)
ax.set_ylabel("Число ошибок на тесте", color=INK_SECONDARY)
ax.set_title(f"Состав ошибок ансамбля (всего {n_total} из 900)", color=INK, fontsize=12, pad=12)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0, colors=INK_SECONDARY)
fig.tight_layout()
fig.savefig("results/error_composition.png", dpi=150, facecolor=SURFACE)
plt.close(fig)
print("\nГрафик сохранён: results/error_composition.png")
