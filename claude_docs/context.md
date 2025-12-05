# Архитектура проекта AI Algotrading

## Tech Stack

| Компонент | Технология |
|-----------|------------|
| Язык | Python 3.10+ |
| Async HTTP | aiohttp |
| Data | pandas, numpy |
| Signals | scipy (argrelextrema) |
| Visualization | matplotlib |
| Config | python-dotenv |
| Binance API | python-binance (для бэктеста) |
| BingX API | Собственный async клиент |

## Архитектурные решения

### 1. Модульность

Проект разделён на независимые модули:


src/core/          # Переиспользуемые компоненты
├── bingx_client   # API клиент (можно заменить на другую биржу)
├── indicators     # Индикаторы (легко добавлять новые)
└── patterns       # Паттерны (Quasimodo, можно добавить другие)

src/strategies/    # Стратегии
├── base           # Абстрактный класс
└── macd_ema_bb    # Конкретная реализация

src/backtest/      # Тестирование
├── engine         # Движок бэктеста
├── optimizer      # Grid Search
└── plots          # Визуализация

src/bot/           # Real-time торговля
└── trader         # Асинхронный бот


### 2. Асинхронность

BingX клиент полностью асинхронный на `aiohttp`:
- Эффективное использование ресурсов
- Поддержка concurrent запросов
- Graceful shutdown по сигналам

### 3. Конфигурация

Все параметры вынесены в `.env`:
- Безопасное хранение API ключей
- Лёгкая смена параметров без изменения кода
- Поддержка CLI аргументов для override

### 4. Расширяемость

Для добавления новой стратегии:

```python
from src.strategies.base import BaseStrategy, StrategyConfig

class MyStrategy(BaseStrategy):
    def check_long_signal(self, df, idx) -> bool:
        # Ваша логика
        pass
    
    def check_short_signal(self, df, idx) -> bool:
        # Ваша логика
        pass
    
    def calculate_sl_tp(self, entry, side, atr):
        # Расчёт SL/TP
        pass
```

## Потоковая диаграмма

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  BingX API  │────▶│  BingxClient │────▶│  DataFrame  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
                                                 ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Signal    │◀────│   Strategy   │◀────│ Indicators  │
└──────┬──────┘     └──────────────┘     └─────────────┘
       │
       ▼
┌─────────────┐     ┌──────────────┐
│   Orders    │────▶│   Position   │
└─────────────┘     └──────────────┘
```

## Безопасность

1. **API ключи** хранятся только в `.env` (игнорируется git)
2. **Testnet по умолчанию** — mainnet требует явного подтверждения
3. **Rate limiting** — паузы между API запросами
4. **Graceful shutdown** — корректное завершение при Ctrl+C
