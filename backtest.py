#!/usr/bin/env python3
"""
Точка входа для бэктестирования.

Использование:
    python backtest.py                           # Быстрый бэктест с дефолтными параметрами
    python backtest.py --symbol BTCUSDT          # Указать пару
    python backtest.py --bars 10000              # Больше данных
    python backtest.py --optimize                # Grid search оптимизация
    python backtest.py --plot                    # Показать графики
"""

import sys
import argparse
import logging
import time
from pathlib import Path

from binance.client import Client as BinanceClient

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent))

from src.core.indicators import calculate_indicators
from src.strategies.base import StrategyConfig
from src.strategies.macd_ema_bb import MacdEmaBbStrategy
from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.optimizer import GridSearchOptimizer, quick_optimize
from src.backtest.plots import plot_equity_curve, plot_combined_report

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def fetch_binance_klines(
    symbol: str,
    interval: str,
    total_bars: int,
    client: BinanceClient = None
) -> pd.DataFrame:
    """
    Получение исторических свечей с Binance Futures.
    
    Args:
        symbol: Торговая пара (например, BTCUSDT)
        interval: Таймфрейм (1m, 5m, 15m, 1h, 4h, 1d)
        total_bars: Общее количество свечей
        client: Binance клиент
    
    Returns:
        DataFrame с OHLCV данными
    """
    if client is None:
        client = BinanceClient()
    
    limit = 1000
    data = []
    end_time = None
    
    logger.info(f"📥 Загрузка {total_bars} свечей {symbol} {interval}...")
    
    while len(data) < total_bars:
        bars_to_fetch = min(limit, total_bars - len(data))
        
        try:
            klines = client.futures_klines(
                symbol=symbol,
                interval=interval,
                limit=bars_to_fetch,
                endTime=end_time
            )
        except Exception as e:
            logger.error(f"Binance API Error: {e}")
            break
        
        if not klines:
            break
        
        data = klines + data
        end_time = klines[0][0] - 1
        time.sleep(0.1)  # Rate limit
    
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[[
        "open", "high", "low", "close", "volume"
    ]].astype(float)
    
    df = df.drop_duplicates("timestamp")
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    logger.info(f"✅ Загружено {len(df)} свечей")
    
    return df


def parse_args():
    """Парсинг аргументов."""
    parser = argparse.ArgumentParser(
        description="AI Algotrading Backtest - Тестирование стратегий"
    )
    
    parser.add_argument(
        "--symbol", "-s",
        type=str,
        default="LTCUSDT",
        help="Торговая пара (default: LTCUSDT)"
    )
    
    parser.add_argument(
        "--interval", "-i",
        type=str,
        default="1h",
        choices=["1m", "5m", "15m", "1h", "4h", "1d"],
        help="Таймфрейм (default: 1h)"
    )
    
    parser.add_argument(
        "--bars", "-b",
        type=int,
        default=5000,
        help="Количество свечей (default: 5000)"
    )
    
    parser.add_argument(
        "--optimize", "-o",
        action="store_true",
        help="Запустить Grid Search оптимизацию"
    )
    
    parser.add_argument(
        "--plot", "-p",
        action="store_true",
        help="Показать графики"
    )
    
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Путь для сохранения графика"
    )
    
    # Параметры стратегии
    parser.add_argument("--ema-period", type=int, default=100)
    parser.add_argument("--bb-std", type=float, default=2.0)
    parser.add_argument("--atr-mult", type=float, default=1.5)
    parser.add_argument("--rr-ratio", type=float, default=2.0)
    
    return parser.parse_args()


def run_single_backtest(args) -> BacktestResult:
    """Запуск одиночного бэктеста."""
    logger.info("=" * 50)
    logger.info("🔬 BACKTEST")
    logger.info("=" * 50)
    
    # Загрузка данных
    df = fetch_binance_klines(
        symbol=args.symbol,
        interval=args.interval,
        total_bars=args.bars
    )
    
    # Конфигурация
    config = StrategyConfig(
        ema_period=args.ema_period,
        bb_std=args.bb_std,
        atr_sl_mult=args.atr_mult,
        rr_ratio=args.rr_ratio,
        use_vol_filter=False,
        squeeze_threshold=1.0,
        use_qml=False
    )
    
    # Расчёт индикаторов
    df = calculate_indicators(
        df,
        ema_period=config.ema_period,
        bb_std=config.bb_std
    )
    
    # Бэктест
    logger.info("📊 Запуск бэктеста...")
    strategy = MacdEmaBbStrategy(config)
    engine = BacktestEngine(strategy, use_trailing=True)
    result = engine.run(df)
    
    # Вывод результатов
    logger.info("\n" + "=" * 50)
    logger.info("📈 РЕЗУЛЬТАТЫ")
    logger.info("=" * 50)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Interval: {args.interval}")
    logger.info(f"Bars: {len(df)}")
    logger.info("-" * 50)
    logger.info(str(result))
    
    # Графики
    if args.plot or args.save_plot:
        plot_combined_report(
            result, df,
            title=f"Backtest: {args.symbol} {args.interval}",
            save_path=args.save_plot
        )
    
    return result


def run_optimization(args):
    """Запуск Grid Search оптимизации."""
    logger.info("=" * 50)
    logger.info("🔍 GRID SEARCH OPTIMIZATION")
    logger.info("=" * 50)
    
    # Функция для получения данных
    def fetch_data(symbol: str, interval: str, limit: int) -> pd.DataFrame:
        return fetch_binance_klines(symbol, interval, limit)
    
    # Сетка параметров
    param_grid = {
        "interval": [args.interval],
        "ema_period": [50, 100, 200],
        "bb_std": [1.5, 2.0, 2.5],
        "atr_sl_mult": [1.0, 1.5, 2.0],
        "rr_ratio": [1.5, 2.0, 2.5, 3.0],
        "use_vol_filter": [False, True],
        "squeeze_threshold": [0.5, 1.0],
        "use_qml": [False],
        "use_qml_extreme_sl": [False]
    }
    
    optimizer = GridSearchOptimizer(
        fetch_data_fn=fetch_data,
        param_grid=param_grid,
        min_trades=10
    )
    
    opt_result = optimizer.run(
        symbol=args.symbol,
        total_bars=args.bars,
        verbose=True
    )
    
    logger.info("\n" + "=" * 50)
    logger.info("🏆 ЛУЧШИЕ ПАРАМЕТРЫ")
    logger.info("=" * 50)
    for key, value in opt_result.best_params.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("-" * 50)
    logger.info(str(opt_result.best_result))
    
    # Графики
    if args.plot or args.save_plot:
        plot_combined_report(
            opt_result.best_result,
            opt_result.best_df,
            title=f"Best Result: {args.symbol}",
            save_path=args.save_plot
        )


def main():
    """Главная функция."""
    args = parse_args()
    
    try:
        if args.optimize:
            run_optimization(args)
        else:
            run_single_backtest(args)
    
    except KeyboardInterrupt:
        logger.info("\n⏹️  Прервано пользователем")
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        raise


if __name__ == "__main__":
    main()
