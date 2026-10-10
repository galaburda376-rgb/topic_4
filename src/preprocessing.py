"""
Этап 2. Предобработка текста.

Метод предобработки (один, как требуется по заданию), состоящий из шагов:
    1. Приведение к нижнему регистру.
    2. Удаление URL и упоминаний (@username) — заменяем на пустоту, они не несут
       лексической информации о теме сообщения.
    3. Удаление пунктуации и эмодзи.
    4. Токенизация (разбиение на слова) регулярным выражением по кириллице/латинице.
    5. Удаление стоп-слов (предлоги, союзы и т.д. — не несут смысла для классификации).
    6. Лемматизация через pymorphy3 (приведение слова к начальной форме:
       "заработок" -> "заработок", "зарабатывать" -> "зарабатывать", "скидкой" -> "скидка").
       Лемматизация нужна, чтобы модель не считала "скидка", "скидкой", "скидки"
       разными признаками — это сильно уменьшает размерность словаря.

Запуск (демонстрация на датасете):
    python src/preprocessing.py

Результат:
    data/dataset_clean.csv — тот же датасет + колонка clean_text
"""

import re
import pandas as pd
import pymorphy3

# Список русских стоп-слов (предлоги, союзы, частицы, местоимения) —
# задаём вручную, чтобы не тянуть загрузку nltk-данных из интернета.
STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все",
    "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по",
    "только", "ее", "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из",
    "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или",
    "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять", "уж", "вам",
    "сказать", "для", "этот", "эта", "эти", "это", "тот", "та", "те", "кто",
    "один", "можно", "при", "наконец", "два", "об", "другой", "хоть", "после",
    "над", "больше", "тот", "через", "эти", "нас", "про", "всего", "них", "какая",
    "много", "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою", "этой",
    "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой", "им", "более",
    "всегда", "конечно", "всю", "ваш", "своей", "который", "там",
}

morph = pymorphy3.MorphAnalyzer()

URL_RE = re.compile(r"(https?://\S+|www\.\S+|t\.me/\S+|bit\.ly/\S+)")
MENTION_RE = re.compile(r"[@#]\w+")
TOKEN_RE = re.compile(r"[а-яёa-z]+", re.IGNORECASE)

_lemma_cache: dict[str, str] = {}


def lemmatize(word: str) -> str:
    if word not in _lemma_cache:
        _lemma_cache[word] = morph.parse(word)[0].normal_form
    return _lemma_cache[word]


def clean_text(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    tokens = TOKEN_RE.findall(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    tokens = [lemmatize(t) for t in tokens]
    return " ".join(tokens)


def main():
    df = pd.read_csv("data/dataset.csv")

    print("Примеры до/после предобработки:\n")
    for text in df["text"].head(5):
        print(f"ДО:    {text}")
        print(f"ПОСЛЕ: {clean_text(text)}\n")

    df["clean_text"] = df["text"].apply(clean_text)
    out_path = "data/dataset_clean.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")

    empty_after_clean = (df["clean_text"].str.len() == 0).sum()
    avg_len_before = df["text"].str.split().str.len().mean()
    avg_len_after = df["clean_text"].str.split().str.len().mean()

    print(f"Сохранено: {out_path}")
    print(f"Среднее число токенов до очистки: {avg_len_before:.1f}")
    print(f"Среднее число токенов после очистки: {avg_len_after:.1f}")
    print(f"Сообщений, опустевших после очистки: {empty_after_clean}")


if __name__ == "__main__":
    main()
