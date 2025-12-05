#!/usr/bin/env python3
"""
Точка входа для real-time торгового бота.

Использование:
    python main.py                    # Запуск с .env конфигурацией
    python main.py --testnet          # Явное указание testnet
    python main.py --symbol BTCUSDT   # Другая пара
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent))

from src.bot.trader import TradingBot, BotConfig
from src.strategies.base import StrategyConfig


def setup_logging(level: str = "INFO"):
    """Настройка логирования."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="AI Algotrading Bot - Real-time торговый бот для BingX"
    )
    
    parser.add_argument(
        "--symbol", "-s",
        type=str,
        default=None,
        help="Торговая пара (например, LTCUSDT)"
    )
    
    parser.add_argument(
        "--interval", "-i",
        type=str,
        default=None,
        choices=["1m", "5m", "15m", "1h", "4h", "1d"],
        help="Таймфрейм"
    )
    
    parser.add_argument(
        "--testnet", "-t",
        action="store_true",
        default=None,
        help="Использовать testnet"
    )
    
    parser.add_argument(
        "--mainnet",
        action="store_true",
        help="Использовать mainnet (ОСТОРОЖНО!)"
    )
    
    parser.add_argument(
        "--leverage", "-l",
        type=int,
        default=None,
        help="Кредитное плечо"
    )
    
    parser.add_argument(
        "--risk", "-r",
        type=float,
        default=None,
        help="Риск на сделку (0.01 = 1%%)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Уровень логирования"
    )
    
    return parser.parse_args()


def load_config(args) -> BotConfig:
    """Загрузка конфигурации из .env и аргументов."""
    
    # Загрузка .env
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Попробуем .env.example
        example_path = Path(__file__).parent / ".env.example"
        if example_path.exists():
            print("⚠️  Файл .env не найден! Скопируйте .env.example в .env и заполните API ключи.")
            sys.exit(1)
    
    # API ключи (обязательные)
    api_key = os.getenv("BINGX_API_KEY", "")
    api_secret = os.getenv("BINGX_API_SECRET", "")
    
    if not api_key or not api_secret or api_key == "your_api_key_here":
        print("❌ Ошибка: API ключи не настроены!")
        print("   Откройте файл .env и заполните BINGX_API_KEY и BINGX_API_SECRET")
        sys.exit(1)
    
    # Параметры из .env (с значениями по умолчанию)
    symbol = args.symbol or os.getenv("SYMBOL", "LTCUSDT")
    interval = args.interval or os.getenv("INTERVAL", "1h")
    leverage = args.leverage or int(os.getenv("LEVERAGE", "20"))
    risk_percent = args.risk or float(os.getenv("RISK_PERCENT", "0.01"))
    
    # Режим testnet/mainnet
    if args.mainnet:
        use_testnet = False
    elif args.testnet:
        use_testnet = True
    else:
        use_testnet = os.getenv("USE_TESTNET", "true").lower() == "true"
    
    # Параметры стратегии из .env
    strategy_config = StrategyConfig(
        ema_period=int(os.getenv("EMA_PERIOD", "100")),
        bb_std=float(os.getenv("BB_STD", "2.0")),
        atr_sl_mult=float(os.getenv("ATR_SL_MULT", "1.5")),
        rr_ratio=float(os.getenv("RR_RATIO", "2.0")),
        qml_order=int(os.getenv("QML_ORDER", "3")),
        qml_recency=int(os.getenv("QML_RECENCY", "20")),
        use_vol_filter=os.getenv("USE_VOL_FILTER", "false").lower() == "true",
        squeeze_threshold=float(os.getenv("SQUEEZE_THRESHOLD", "1.0")),
        use_qml=False,
        use_qml_extreme_sl=os.getenv("USE_QML_EXTREME_SL", "false").lower() == "true"
    )
    
    return BotConfig(
        api_key=api_key,
        api_secret=api_secret,
        symbol=symbol,
        interval=interval,
        leverage=leverage,
        risk_percent=risk_percent,
        use_testnet=use_testnet,
        strategy_config=strategy_config
    )


async def main():
    """Главная функция."""
    args = parse_args()
    setup_logging(args.log_level)
    
    config = load_config(args)
    
    # Предупреждение для mainnet
    if not config.use_testnet:
        print("\n" + "⚠️ " * 20)
        print("⚠️  ВНИМАНИЕ: Вы запускаете бота на MAINNET!")
        print("⚠️  Это означает РЕАЛЬНЫЕ деньги!")
        print("⚠️ " * 20 + "\n")
        
        response = input("Вы уверены? (yes/no): ")
        if response.lower() != "yes":
            print("Отменено.")
            sys.exit(0)
    
    # Запуск бота
    bot = TradingBot(config)
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        print("\n⏹️  Прервано пользователем")


if __name__ == "__main__":
    asyncio.run(main())
