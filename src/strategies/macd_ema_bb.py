"""
Стратегия MACD + EMA + Bollinger Bands + QML.
Основана на видео: https://www.youtube.com/watch?v=OWsum6xcNvM
"""

from typing import Optional, Tuple

import pandas as pd

from .base import BaseStrategy, StrategyConfig, Signal
from ..core.indicators import (
    check_macd_cross_up,
    check_macd_cross_down,
    check_price_above_ema,
    check_price_below_ema
)
from ..core.patterns import detect_qml_bull, detect_qml_bear


class MacdEmaBbStrategy(BaseStrategy):
    """
    Стратегия на основе MACD, EMA и Bollinger Bands.
    
    Условия входа в Long:
    1. MACD пересекает сигнальную линию снизу вверх
    2. Цена выше EMA (тренд вверх)
    3. (Опционально) Обнаружен бычий Quasimodo паттерн
    4. Фильтры: объём выше среднего, нет squeeze
    
    Условия входа в Short:
    1. MACD пересекает сигнальную линию сверху вниз
    2. Цена ниже EMA (тренд вниз)
    3. (Опционально) Обнаружен медвежий Quasimodo паттерн
    4. Фильтры: объём выше среднего, нет squeeze
    
    Стоп-лосс: ATR × multiplier (или QML экстремум)
    Тейк-профит: Risk × R:R ratio
    """
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config)
        self._last_qml_bull: Optional[float] = None
        self._last_qml_bear: Optional[float] = None
    
    def check_long_signal(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Проверка условий для входа в long.
        
        Args:
            df: DataFrame с индикаторами
            idx: Текущий индекс свечи
        
        Returns:
            True если все условия long выполнены
        """
        # 1. MACD пересечение вверх
        if not check_macd_cross_up(df["macd"], df["signal"], idx):
            return False
        
        # 2. Цена выше EMA (аптренд)
        if not check_price_above_ema(df["close"], df["ema"], idx):
            return False
        
        # 3. QML паттерн (опционально)
        if self.config.use_qml:
            qml_level = detect_qml_bull(
                df, idx,
                order=self.config.qml_order,
                recency=self.config.qml_recency
            )
            if qml_level is None:
                return False
            self._last_qml_bull = qml_level
        
        return True
    
    def check_short_signal(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Проверка условий для входа в short.
        
        Args:
            df: DataFrame с индикаторами
            idx: Текущий индекс свечи
        
        Returns:
            True если все условия short выполнены
        """
        # 1. MACD пересечение вниз
        if not check_macd_cross_down(df["macd"], df["signal"], idx):
            return False
        
        # 2. Цена ниже EMA (даунтренд)
        if not check_price_below_ema(df["close"], df["ema"], idx):
            return False
        
        # 3. QML паттерн (опционально)
        if self.config.use_qml:
            qml_level = detect_qml_bear(
                df, idx,
                order=self.config.qml_order,
                recency=self.config.qml_recency
            )
            if qml_level is None:
                return False
            self._last_qml_bear = qml_level
        
        return True
    
    def calculate_sl_tp(
        self,
        entry_price: float,
        side: str,
        atr: float,
        qml_level: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Расчёт стоп-лосса и тейк-профита.
        
        Для Long:
            SL = Entry - (ATR × mult) или QML_low - (ATR × mult)
            TP = Entry + (Risk × R:R)
        
        Для Short:
            SL = Entry + (ATR × mult) или QML_high + (ATR × mult)
            TP = Entry - (Risk × R:R)
        
        Args:
            entry_price: Цена входа
            side: 'long' или 'short'
            atr: Текущее значение ATR
            qml_level: Уровень QML экстремума
        
        Returns:
            Tuple (stop_loss, take_profit)
        """
        config = self.config
        atr_offset = atr * config.atr_sl_mult
        
        if side == "long":
            # Стоп-лосс
            if config.use_qml_extreme_sl and self._last_qml_bull is not None:
                sl = self._last_qml_bull - atr_offset
            else:
                sl = entry_price - atr_offset
            
            # Тейк-профит
            risk = entry_price - sl
            tp = entry_price + risk * config.rr_ratio
        
        else:  # short
            # Стоп-лосс
            if config.use_qml_extreme_sl and self._last_qml_bear is not None:
                sl = self._last_qml_bear + atr_offset
            else:
                sl = entry_price + atr_offset
            
            # Тейк-профит
            risk = sl - entry_price
            tp = entry_price - risk * config.rr_ratio
        
        return sl, tp
    
    def check_exit_conditions(
        self,
        df: pd.DataFrame,
        idx: int,
        position_side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float
    ) -> Tuple[bool, Optional[float], str]:
        """
        Проверка условий выхода из позиции.
        
        Условия выхода:
        1. Достижение стоп-лосса
        2. Достижение тейк-профита
        3. Обратное пересечение MACD
        4. Касание противоположной полосы Боллинджера
        
        Args:
            df: DataFrame с индикаторами
            idx: Текущий индекс свечи
            position_side: 'long' или 'short'
            entry_price: Цена входа
            stop_loss: Уровень стоп-лосса
            take_profit: Уровень тейк-профита
        
        Returns:
            Tuple (should_exit, exit_price, exit_reason)
        """
        high = df["high"].iloc[idx]
        low = df["low"].iloc[idx]
        close = df["close"].iloc[idx]
        
        if position_side == "long":
            # Проверка SL
            if low <= stop_loss:
                return True, stop_loss, "stop_loss"
            
            # Проверка TP
            if high >= take_profit:
                return True, take_profit, "take_profit"
            
            # Обратное пересечение MACD
            if check_macd_cross_down(df["macd"], df["signal"], idx):
                return True, close, "macd_reverse"
            
            # Касание верхней полосы Боллинджера
            if high >= df["bb_upper"].iloc[idx]:
                return True, df["bb_upper"].iloc[idx], "bb_touch"
        
        else:  # short
            # Проверка SL
            if high >= stop_loss:
                return True, stop_loss, "stop_loss"
            
            # Проверка TP
            if low <= take_profit:
                return True, take_profit, "take_profit"
            
            # Обратное пересечение MACD
            if check_macd_cross_up(df["macd"], df["signal"], idx):
                return True, close, "macd_reverse"
            
            # Касание нижней полосы Боллинджера
            if low <= df["bb_lower"].iloc[idx]:
                return True, df["bb_lower"].iloc[idx], "bb_touch"
        
        return False, None, ""
    
    def update_trailing_stop(
        self,
        df: pd.DataFrame,
        idx: int,
        position_side: str,
        entry_price: float,
        current_sl: float,
        reached_1to1: bool
    ) -> Tuple[float, bool]:
        """
        Обновление trailing stop на основе Bollinger Mid после достижения 1:1.
        
        Args:
            df: DataFrame с индикаторами
            idx: Текущий индекс свечи
            position_side: 'long' или 'short'
            entry_price: Цена входа
            current_sl: Текущий стоп-лосс
            reached_1to1: Достигнут ли уровень 1:1
        
        Returns:
            Tuple (new_stop_loss, new_reached_1to1)
        """
        high = df["high"].iloc[idx]
        low = df["low"].iloc[idx]
        bb_mid = df["bb_mid"].iloc[idx]
        
        new_reached = reached_1to1
        new_sl = current_sl
        
        if position_side == "long":
            risk = entry_price - current_sl
            target_1to1 = entry_price + risk
            
            # Проверка достижения 1:1
            if not reached_1to1 and high >= target_1to1:
                new_reached = True
            
            # Trailing по BB mid после 1:1
            if new_reached:
                new_sl = max(current_sl, bb_mid)
        
        else:  # short
            risk = current_sl - entry_price
            target_1to1 = entry_price - risk
            
            # Проверка достижения 1:1
            if not reached_1to1 and low <= target_1to1:
                new_reached = True
            
            # Trailing по BB mid после 1:1
            if new_reached:
                new_sl = min(current_sl, bb_mid)
        
        return new_sl, new_reached
