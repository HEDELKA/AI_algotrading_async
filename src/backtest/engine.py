"""
Движок бэктестирования торговых стратегий.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np

from ..strategies.base import BaseStrategy, StrategyConfig
from ..core.indicators import calculate_indicators


@dataclass
class Trade:
    """Информация о сделке."""
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    side: str
    pnl_pct: float
    exit_reason: str


@dataclass
class BacktestResult:
    """Результаты бэктеста."""
    total_pnl_pct: float
    win_rate: float
    num_trades: int
    avg_pnl: float
    max_drawdown: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "total_pnl_pct": self.total_pnl_pct,
            "win_rate": self.win_rate,
            "num_trades": self.num_trades,
            "avg_pnl": self.avg_pnl,
            "max_drawdown": self.max_drawdown
        }
    
    def __str__(self) -> str:
        return (
            f"Total PNL: {self.total_pnl_pct:.2f}%\n"
            f"Win Rate: {self.win_rate:.2f}%\n"
            f"Trades: {self.num_trades}\n"
            f"Avg PNL: {self.avg_pnl:.2f}%\n"
            f"Max DD: {self.max_drawdown:.2f}%"
        )


class BacktestEngine:
    """
    Движок бэктестирования.
    
    Особенности:
    - Поддержка любой стратегии, наследующей BaseStrategy
    - Trailing stop по Bollinger Mid после 1:1
    - Множественные условия выхода
    - Расчёт equity curve и max drawdown
    """
    
    def __init__(
        self,
        strategy: BaseStrategy,
        use_trailing: bool = True
    ):
        """
        Инициализация движка.
        
        Args:
            strategy: Экземпляр торговой стратегии
            use_trailing: Использовать trailing stop
        """
        self.strategy = strategy
        self.use_trailing = use_trailing
    
    def run(
        self,
        df: pd.DataFrame,
        max_lookback: int = 100
    ) -> BacktestResult:
        """
        Запуск бэктеста.
        
        Args:
            df: DataFrame со свечами (должен содержать индикаторы)
            max_lookback: Минимальное количество свечей для прогрева индикаторов
        
        Returns:
            BacktestResult с результатами
        """
        trades: List[Trade] = []
        equity_curve: List[float] = [1.0]
        current_equity = 1.0
        
        in_position = False
        position: Optional[Dict[str, Any]] = None
        
        for i in range(max_lookback, len(df)):
            if not in_position:
                # Проверка сигнала на вход
                signal = self.strategy.generate_signal(df, i)
                
                if signal is not None:
                    position = {
                        "entry_idx": i,
                        "entry_price": signal.entry_price,
                        "sl": signal.stop_loss,
                        "tp": signal.take_profit,
                        "side": signal.side,
                        "reached_1to1": False
                    }
                    in_position = True
            
            else:
                # Управление позицией
                pos = position
                
                # Trailing stop
                if self.use_trailing:
                    new_sl, new_reached = self.strategy.update_trailing_stop(
                        df, i,
                        pos["side"],
                        pos["entry_price"],
                        pos["sl"],
                        pos["reached_1to1"]
                    )
                    pos["sl"] = new_sl
                    pos["reached_1to1"] = new_reached
                
                # Проверка выхода
                should_exit, exit_price, exit_reason = self.strategy.check_exit_conditions(
                    df, i,
                    pos["side"],
                    pos["entry_price"],
                    pos["sl"],
                    pos["tp"]
                )
                
                if should_exit:
                    # Расчёт PNL
                    if pos["side"] == "long":
                        pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
                    else:
                        pnl_pct = (pos["entry_price"] - exit_price) / pos["entry_price"] * 100
                    
                    # Сохранение сделки
                    trade = Trade(
                        entry_idx=pos["entry_idx"],
                        exit_idx=i,
                        entry_price=pos["entry_price"],
                        exit_price=exit_price,
                        side=pos["side"],
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason
                    )
                    trades.append(trade)
                    
                    # Обновление equity
                    current_equity *= (1 + pnl_pct / 100)
                    equity_curve.append(current_equity)
                    
                    # Сброс позиции
                    in_position = False
                    position = None
        
        # Расчёт статистики
        return self._calculate_stats(trades, equity_curve)
    
    def _calculate_stats(
        self,
        trades: List[Trade],
        equity_curve: List[float]
    ) -> BacktestResult:
        """Расчёт статистики бэктеста."""
        if not trades:
            return BacktestResult(
                total_pnl_pct=0.0,
                win_rate=0.0,
                num_trades=0,
                avg_pnl=0.0,
                max_drawdown=0.0,
                trades=[],
                equity_curve=[1.0]
            )
        
        pnls = [t.pnl_pct for t in trades]
        total_pnl = sum(pnls)
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        avg_pnl = np.mean(pnls)
        
        # Max Drawdown
        equity_arr = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_arr)
        drawdowns = (running_max - equity_arr) / running_max * 100
        max_drawdown = np.max(drawdowns)
        
        return BacktestResult(
            total_pnl_pct=total_pnl,
            win_rate=win_rate,
            num_trades=len(trades),
            avg_pnl=avg_pnl,
            max_drawdown=max_drawdown,
            trades=trades,
            equity_curve=equity_curve
        )


def run_backtest(
    df: pd.DataFrame,
    config: Optional[StrategyConfig] = None,
    use_trailing: bool = True
) -> BacktestResult:
    """
    Упрощённая функция для запуска бэктеста.
    
    Args:
        df: DataFrame со свечами (OHLCV)
        config: Конфигурация стратегии
        use_trailing: Использовать trailing stop
    
    Returns:
        BacktestResult с результатами
    """
    from ..strategies.macd_ema_bb import MacdEmaBbStrategy
    
    config = config or StrategyConfig()
    
    # Расчёт индикаторов
    df = calculate_indicators(
        df,
        ema_period=config.ema_period,
        macd_fast=config.macd_fast,
        macd_slow=config.macd_slow,
        macd_signal=config.macd_signal,
        bb_period=config.bb_period,
        bb_std=config.bb_std,
        atr_period=config.atr_period,
        vol_period=config.vol_period
    )
    
    # Создание стратегии и движка
    strategy = MacdEmaBbStrategy(config)
    engine = BacktestEngine(strategy, use_trailing)
    
    return engine.run(df)
