"""
Этап 3. Разбиение датасета на train/val/test.

Зачем отдельный этап: и baseline-модель (TF-IDF), и нейросети (RNN, CNN) должны
обучаться и проверяться на одних и тех же выборках, иначе сравнение методов
будет нечестным. val (валидация) используется для подбора весов ансамбля,
test — только для финальной оценки, его модели не видят во время обучения.

Разбиение стратифицированное (сохраняет пропорцию spam/clickbait в каждой части).

Запуск:
    python src/split_data.py

Результат:
    data/train.csv (70%), data/val.csv (15%), data/test.csv (15%)
"""

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def main():
    df = pd.read_csv("data/dataset_clean.csv")

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=RANDOM_STATE
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=RANDOM_STATE
    )

    train_df.to_csv("data/train.csv", index=False, encoding="utf-8")
    val_df.to_csv("data/val.csv", index=False, encoding="utf-8")
    test_df.to_csv("data/test.csv", index=False, encoding="utf-8")

    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = part["label"].value_counts().to_dict()
        print(f"{name}: {len(part)} строк, {counts}")


if __name__ == "__main__":
    main()
