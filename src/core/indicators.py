"""
Технические индикаторы для торговых стратегий.
"""

import numpy as np
import pandas as pd
from typing import Optional


def calculate_ema(
    series: pd.Series,
    period: int = 50
) -> pd.Series:
    """
    Расчёт экспоненциальной скользящей средней (EMA).
    
    Args:
        series: Временной ряд цен
        period: Период сглаживания
    
    Returns:
        Series с значениями EMA
    """
    return series.ewm(span=period, adjust=False).mean()


def calculate_sma(
    series: pd.Series,
    period: int = 20
) -> pd.Series:
    """
    Расчёт простой скользящей средней (SMA).
    
    Args:
        series: Временной ряд цен
        period: Период сглаживания
    
    Returns:
        Series с значениями SMA
    """
    return series.rolling(period).mean()


def calculate_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Расчёт MACD (Moving Average Convergence Divergence).
    
    Args:
        series: Временной ряд цен закрытия
        fast_period: Период быстрой EMA
        slow_period: Период медленной EMA
        signal_period: Период сигнальной линии
    
    Returns:
        Tuple: (macd_line, signal_line, histogram)
    """
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Расчёт линий Боллинджера.
    
    Args:
        series: Временной ряд цен
        period: Период SMA
        std_dev: Множитель стандартного отклонения
    
    Returns:
        Tuple: (upper_band, middle_band, lower_band)
    """
    middle = series.rolling(period).mean()
    std = series.rolling(period).std()
    
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    
    return upper, middle, lower


def calculate_bollinger_width(
    upper: pd.Series,
    middle: pd.Series,
    lower: pd.Series
) -> pd.Series:
    """
    Расчёт ширины полос Боллинджера в процентах.
    Используется для детекции сжатия (squeeze).
    
    Args:
        upper: Верхняя полоса
        middle: Средняя линия
        lower: Нижняя полоса
    
    Returns:
        Series с шириной в процентах
    """
    return ((upper - lower) / middle) * 100


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Расчёт Average True Range (ATR).
    
    Args:
        high: Максимумы свечей
        low: Минимумы свечей
        close: Цены закрытия
        period: Период сглаживания
    
    Returns:
        Series с значениями ATR
    """
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = true_range.rolling(period).mean()
    
    return atr


def calculate_rsi(
    series: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Расчёт RSI (Relative Strength Index).
    
    Args:
        series: Временной ряд цен
        period: Период расчёта
    
    Returns:
        Series с значениями RSI (0-100)
    """
    delta = series.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_volume_sma(
    volume: pd.Series,
    period: int = 20
) -> pd.Series:
    """
    Расчёт средней объёма.
    
    Args:
        volume: Объёмы свечей
        period: Период сглаживания
    
    Returns:
        Series со средним объёмом
    """
    return volume.rolling(period).mean()


def calculate_indicators(
    df: pd.DataFrame,
    ema_period: int = 100,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_period: int = 20,
    bb_std: float = 2.0,
    atr_period: int = 14,
    vol_period: int = 20
) -> pd.DataFrame:
    """
    Расчёт всех индикаторов для стратегии MACD+EMA+BB.
    
    Args:
        df: DataFrame со свечами (OHLCV)
        ema_period: Период EMA для тренда
        macd_fast: Быстрый период MACD
        macd_slow: Медленный период MACD
        macd_signal: Сигнальный период MACD
        bb_period: Период Bollinger Bands
        bb_std: Множитель стандартного отклонения BB
        atr_period: Период ATR
        vol_period: Период для среднего объёма
    
    Returns:
        DataFrame с добавленными индикаторами
    """
    df = df.copy()
    
    # EMA для определения тренда
    df["ema"] = calculate_ema(df["close"], ema_period)
    
    # MACD для сигналов
    df["macd"], df["signal"], df["macd_hist"] = calculate_macd(
        df["close"], macd_fast, macd_slow, macd_signal
    )
    
    # Bollinger Bands для волатильности и выходов
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = calculate_bollinger_bands(
        df["close"], bb_period, bb_std
    )
    
    # Ширина полос для детекции squeeze
    df["bb_width"] = calculate_bollinger_width(
        df["bb_upper"], df["bb_mid"], df["bb_lower"]
    )
    
    # ATR для расчёта стоп-лосса
    df["atr"] = calculate_atr(
        df["high"], df["low"], df["close"], atr_period
    )
    
    # Средний объём для фильтра
    df["vol_avg"] = calculate_volume_sma(df["volume"], vol_period)
    
    return df


def check_macd_cross_up(macd: pd.Series, signal: pd.Series, idx: int) -> bool:
    """Проверка пересечения MACD вверх."""
    if idx < 1:
        return False
    return macd.iloc[idx] > signal.iloc[idx] and macd.iloc[idx - 1] <= signal.iloc[idx - 1]


def check_macd_cross_down(macd: pd.Series, signal: pd.Series, idx: int) -> bool:
    """Проверка пересечения MACD вниз."""
    if idx < 1:
        return False
    return macd.iloc[idx] < signal.iloc[idx] and macd.iloc[idx - 1] >= signal.iloc[idx - 1]


def check_price_above_ema(close: pd.Series, ema: pd.Series, idx: int) -> bool:
    """Проверка: цена выше EMA."""
    return close.iloc[idx] > ema.iloc[idx]


def check_price_below_ema(close: pd.Series, ema: pd.Series, idx: int) -> bool:
    """Проверка: цена ниже EMA."""
    return close.iloc[idx] < ema.iloc[idx]


def check_volume_filter(
    volume: pd.Series,
    vol_avg: pd.Series,
    idx: int,
    multiplier: float = 0.8
) -> bool:
    """Проверка: объём выше среднего × multiplier."""
    return volume.iloc[idx] > vol_avg.iloc[idx] * multiplier


def check_no_squeeze(bb_width: pd.Series, idx: int, threshold: float = 1.0) -> bool:
    """Проверка отсутствия сжатия полос Боллинджера."""
    return bb_width.iloc[idx] >= threshold
