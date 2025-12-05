"""
Grid Search оптимизация параметров стратегии.
"""

import itertools
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable

import pandas as pd

from .engine import BacktestEngine, BacktestResult
from ..strategies.base import StrategyConfig
from ..strategies.macd_ema_bb import MacdEmaBbStrategy
from ..core.indicators import calculate_indicators

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Результат оптимизации."""
    best_params: Dict[str, Any]
    best_result: BacktestResult
    best_df: pd.DataFrame
    all_results: List[Dict[str, Any]]


class GridSearchOptimizer:
    """
    Grid Search оптимизатор для параметров стратегии.
    
    Перебирает все комбинации параметров и находит лучшую
    на основе заданной метрики (по умолчанию total_pnl_pct).
    """
    
    # Параметры по умолчанию для перебора
    DEFAULT_PARAM_GRID = {
        "interval": ["1h"],
        "ema_period": [50, 100, 200],
        "bb_std": [1.5, 2.0, 2.5],
        "atr_sl_mult": [1.0, 1.5, 2.0],
        "rr_ratio": [1.5, 2.0, 2.5],
        "qml_order": [3, 5],
        "qml_recency": [15, 20, 30],
        "use_vol_filter": [False, True],
        "squeeze_threshold": [0.5, 1.0, 1.5],
        "use_qml": [False],
        "use_qml_extreme_sl": [False]
    }
    
    def __init__(
        self,
        fetch_data_fn: Callable[[str, str, int], pd.DataFrame],
        param_grid: Optional[Dict[str, List[Any]]] = None,
        min_trades: int = 10,
        metric: str = "total_pnl_pct"
    ):
        """
        Инициализация оптимизатора.
        
        Args:
            fetch_data_fn: Функция для получения данных (symbol, interval, limit) -> DataFrame
            param_grid: Словарь параметров для перебора
            min_trades: Минимальное количество сделок для валидного результата
            metric: Метрика для оптимизации
        """
        self.fetch_data_fn = fetch_data_fn
        self.param_grid = param_grid or self.DEFAULT_PARAM_GRID
        self.min_trades = min_trades
        self.metric = metric
    
    def _params_to_config(self, params: Dict[str, Any]) -> StrategyConfig:
        """Преобразование словаря параметров в StrategyConfig."""
        return StrategyConfig(
            ema_period=params.get("ema_period", 100),
            bb_std=params.get("bb_std", 2.0),
            atr_sl_mult=params.get("atr_sl_mult", 1.5),
            rr_ratio=params.get("rr_ratio", 2.0),
            qml_order=params.get("qml_order", 3),
            qml_recency=params.get("qml_recency", 20),
            use_vol_filter=params.get("use_vol_filter", False),
            squeeze_threshold=params.get("squeeze_threshold", 1.0),
            use_qml=params.get("use_qml", False),
            use_qml_extreme_sl=params.get("use_qml_extreme_sl", False)
        )
    
    def _calculate_score(self, result: BacktestResult, params: Dict[str, Any]) -> float:
        """
        Расчёт скора для ранжирования результатов.
        
        Штраф за малое количество сделок.
        """
        if result.num_trades < self.min_trades:
            return getattr(result, self.metric) - 1000  # Большой штраф
        
        return getattr(result, self.metric)
    
    def run(
        self,
        symbol: str = "BTCUSDT",
        total_bars: int = 5000,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Запуск Grid Search оптимизации.
        
        Args:
            symbol: Торговая пара
            total_bars: Количество свечей для тестирования
            verbose: Выводить прогресс
        
        Returns:
            OptimizationResult с лучшими параметрами
        """
        keys = list(self.param_grid.keys())
        combos = list(itertools.product(*self.param_grid.values()))
        
        best_result: Optional[BacktestResult] = None
        best_params: Optional[Dict[str, Any]] = None
        best_score = float("-inf")
        best_df: Optional[pd.DataFrame] = None
        all_results: List[Dict[str, Any]] = []
        
        total_combos = len(combos)
        
        for i, combo in enumerate(combos, 1):
            params = dict(zip(keys, combo))
            
            if verbose:
                logger.info(f"[{i}/{total_combos}] Testing: {params}")
            
            try:
                # Получение данных
                interval = params.get("interval", "1h")
                df = self.fetch_data_fn(symbol, interval, total_bars)
                
                # Расчёт индикаторов
                config = self._params_to_config(params)
                df = calculate_indicators(
                    df,
                    ema_period=config.ema_period,
                    bb_std=config.bb_std,
                    atr_period=config.atr_period,
                    vol_period=config.vol_period
                )
                
                # Бэктест
                strategy = MacdEmaBbStrategy(config)
                engine = BacktestEngine(strategy, use_trailing=True)
                result = engine.run(df)
                
                # Расчёт скора
                score = self._calculate_score(result, params)
                
                # Сохранение результата
                all_results.append({
                    "params": params,
                    "score": score,
                    **result.to_dict()
                })
                
                if verbose:
                    logger.info(
                        f"  PNL: {result.total_pnl_pct:.2f}%, "
                        f"WR: {result.win_rate:.2f}%, "
                        f"Trades: {result.num_trades}"
                    )
                
                # Обновление лучшего
                if score > best_score:
                    best_score = score
                    best_result = result
                    best_params = params
                    best_df = df.copy()
            
            except Exception as e:
                logger.error(f"Error testing params {params}: {e}")
                continue
        
        if best_result is None:
            raise ValueError("No valid results found during optimization")
        
        if verbose:
            logger.info("\n" + "=" * 50)
            logger.info("BEST PARAMETERS:")
            for key, value in best_params.items():
                logger.info(f"  {key}: {value}")
            logger.info("-" * 50)
            logger.info(str(best_result))
        
        return OptimizationResult(
            best_params=best_params,
            best_result=best_result,
            best_df=best_df,
            all_results=all_results
        )


def quick_optimize(
    fetch_data_fn: Callable[[str, str, int], pd.DataFrame],
    symbol: str = "LTCUSDT",
    total_bars: int = 5000
) -> OptimizationResult:
    """
    Быстрая оптимизация с уменьшенной сеткой параметров.
    
    Args:
        fetch_data_fn: Функция для получения данных
        symbol: Торговая пара
        total_bars: Количество свечей
    
    Returns:
        OptimizationResult
    """
    quick_grid = {
        "interval": ["1h"],
        "ema_period": [100],
        "bb_std": [2.0],
        "atr_sl_mult": [1.5],
        "rr_ratio": [2.0],
        "qml_order": [3],
        "qml_recency": [20],
        "use_vol_filter": [False],
        "squeeze_threshold": [1.0],
        "use_qml": [False],
        "use_qml_extreme_sl": [False]
    }
    
    optimizer = GridSearchOptimizer(
        fetch_data_fn=fetch_data_fn,
        param_grid=quick_grid
    )
    
    return optimizer.run(symbol=symbol, total_bars=total_bars)
