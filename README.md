# Классификация спама и кликбейта в Telegram-сообщениях

Учебный проект по курсу «Обработка естественного языка» (магистратура).

Кейс: разработать систему, классифицирующую сообщения из Telegram на **spam**
(нежелательные сообщения — реклама, мошенничество, автоматика) и **clickbait**
(заманчивые заголовки ради кликов).

Полный отчёт по заданию: [REPORT.md](REPORT.md).

## Структура проекта

```
data/
  generate_dataset.py   — генератор синтетического датасета (этап 1)
  dataset.csv            — сырой датасет (сообщение, метка, флаги шума)
  dataset_clean.csv      — датасет после предобработки (+ колонка clean_text)
  train.csv / val.csv / test.csv — разбиение 70/15/15

src/
  preprocessing.py        — предобработка текста (этап 2)
  split_data.py            — разбиение на train/val/test (этап 3)
  train_baseline.py        — TF-IDF + логистическая регрессия (этап 4)
  train_rnn_cnn_ensemble.py — ансамбль RNN+CNN с весами по валидации (этап 5, бонус)
  error_analysis.py        — анализ ошибок классификации (этап 6)

results/   — метрики, графики, CSV с ошибками (генерируются скриптами выше)
```

## Как воспроизвести результаты

```bash
pip install -r requirements.txt

python data/generate_dataset.py
python src/preprocessing.py
python src/split_data.py
python src/train_baseline.py
python src/train_rnn_cnn_ensemble.py
python src/error_analysis.py
```

На Windows перед запуском в PowerShell выставьте `$env:PYTHONUTF8=1`, иначе
кириллица в консоли может ломаться.
