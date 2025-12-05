# Команды проекта AI Algotrading

## Установка

```bash
# Клонирование (если нужно)
git clone <repo>
cd AI_algotrading

# Виртуальное окружение (рекомендуется)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt

# Настройка конфигурации
cp .env.example .env
nano .env  # заполнить API ключи
```

## Бэктест

```bash
# Быстрый тест с дефолтными параметрами
python backtest.py

# Указать пару и количество свечей
python backtest.py --symbol BTCUSDT --bars 10000

# С графиками
python backtest.py --plot

# Сохранить график в файл
python backtest.py --plot --save-plot results/report.png

# Grid Search оптимизация
python backtest.py --optimize

# Полный пример
python backtest.py \
    --symbol LTCUSDT \
    --interval 1h \
    --bars 5000 \
    --ema-period 100 \
    --bb-std 2.0 \
    --rr-ratio 2.0 \
    --plot
```

## Real-time Bot

```bash
# Запуск на testnet (по умолчанию)
python main.py

# Явное указание testnet
python main.py --testnet

# Указать пару
python main.py --symbol BTCUSDT

# Изменить таймфрейм
python main.py --interval 4h

# Изменить плечо и риск
python main.py --leverage 10 --risk 0.02

# MAINNET (ОСТОРОЖНО!)
python main.py --mainnet

# Debug логирование
python main.py --log-level DEBUG
```

## Troubleshooting

### Ошибка импорта модулей
```bash
# Убедитесь, что вы в корне проекта
cd /path/to/AI_algotrading
python backtest.py
```

### Ошибка API ключей
```bash
# Проверьте .env файл
cat .env | grep BINGX

# Убедитесь, что ключи без кавычек
BINGX_API_KEY=abc123  # правильно
BINGX_API_KEY="abc123"  # неправильно
```

### Binance rate limit
```bash
# Уменьшите количество свечей
python backtest.py --bars 1000

# Или увеличьте паузу в fetch_binance_klines
```

### BingX API ошибки
```bash
# Проверьте время сервера
# Бот автоматически синхронизирует, но могут быть проблемы

# Включите debug логирование
python main.py --log-level DEBUG
```

## Git

```bash
# Проверить что .env не в коммите
git status

# Добавить изменения (кроме .env)
git add .
git commit -m "Update strategy"
git push
```
