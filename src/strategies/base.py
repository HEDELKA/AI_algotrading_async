"""
Базовый класс для торговых стратегий.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


@dataclass
class Signal:
    """Торговый сигнал."""
    side: str  # 'long' или 'short'
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: Optional[float] = None
    
    @property
    def risk(self) -> float:
        """Риск на сделку (расстояние до SL)."""
        if self.side == "long":
            return self.entry_price - self.stop_loss
        return self.stop_loss - self.entry_price
    
    @property
    def reward(self) -> float:
        """Потенциальная прибыль (расстояние до TP)."""
        if self.side == "long":
            return self.take_profit - self.entry_price
        return self.entry_price - self.take_profit
    
    @property
    def rr_ratio(self) -> float:
        """Соотношение Risk/Reward."""
        if self.risk == 0:
            return 0
        return self.reward / self.risk


@dataclass
class StrategyConfig:
    """Конфигурация стратегии."""
    # EMA
    ema_period: int = 100
    
    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    
    # ATR
    atr_period: int = 14
    atr_sl_mult: float = 1.5
    
    # Risk/Reward
    rr_ratio: float = 2.0
    
    # QML
    qml_order: int = 3
    qml_recency: int = 20
    use_qml: bool = False
    use_qml_extreme_sl: bool = False
    
    # Фильтры
    use_vol_filter: bool = False
    vol_filter_mult: float = 0.8
    squeeze_threshold: float = 1.0
    
    # Volume
    vol_period: int = 20


class BaseStrategy(ABC):
    """Абстрактный базовый класс для торговых стратегий."""
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        """
        Инициализация стратегии.
        
        Args:
            config: Конфигурация параметров стратегии
        """
        self.config = config or StrategyConfig()
    
    @abstractmethod
    def check_long_signal(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Проверка условий для входа в long.
        
        Args:
            df: DataFrame с индикаторами
            idx: Текущий индекс свечи
        
        Returns:
            True если все условия выполнены
        """
        pass
    
    @abstractmethod
    def check_short_signal(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Проверка условий для входа в short.
        
        Args:
            df: DataFrame с индикаторами
            idx: Текущий индекс свечи
        
        Returns:
            True если все условия выполнены
        """
        pass
    
    @abstractmethod
    def calculate_sl_tp(
        self,
        entry_price: float,
        side: str,
        atr: float,
        qml_level: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Расчёт стоп-лосса и тейк-профита.
        
        Args:
            entry_price: Цена входа
            side: 'long' или 'short'
            atr: Текущее значение ATR
            qml_level: Уровень QML экстремума (опционально)
        
        Returns:
            Tuple (stop_loss, take_profit)
        """
        pass
    
    def check_filters(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Проверка общих фильтров (объём, squeeze).
        
        Args:
            df: DataFrame с индикаторами
            idx: Текущий индекс свечи
        
        Returns:
            True если все фильтры пройдены
        """
        config = self.config
        
        # Фильтр объёма
        if config.use_vol_filter:
            if df["volume"].iloc[idx] <= df["vol_avg"].iloc[idx] * config.vol_filter_mult:
                return False
        
        # Фильтр squeeze (узкие полосы Боллинджера = низкая волатильность)
        if df["bb_width"].iloc[idx] < config.squeeze_threshold:
            return False
        
        return True
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        idx: int,
        current_price: Optional[float] = None
    ) -> Optional[Signal]:
        """
        Генерация торгового сигнала.
        
        Args:
            df: DataFrame с индикаторами
            idx: Текущий индекс свечи
            current_price: Текущая цена (для live trading)
        
        Returns:
            Signal объект или None
        """
        # Проверка фильтров
        if not self.check_filters(df, idx):
            return None
        
        entry_price = current_price or df["close"].iloc[idx]
        atr = df["atr"].iloc[idx]
        
        # Проверка long
        if self.check_long_signal(df, idx):
            sl, tp = self.calculate_sl_tp(entry_price, "long", atr)
            return Signal(
                side="long",
                entry_price=entry_price,
                stop_loss=sl,
                take_profit=tp
            )
        
        # Проверка short
        if self.check_short_signal(df, idx):
            sl, tp = self.calculate_sl_tp(entry_price, "short", atr)
            return Signal(
                side="short",
                entry_price=entry_price,
                stop_loss=sl,
                take_profit=tp
            )
        
        return None
