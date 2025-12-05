"""
Детекция паттернов Price Action.
Quasimodo (QML) - паттерн разворота на основе структуры экстремумов.
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import Optional


def detect_qml_bull(
    df: pd.DataFrame,
    idx: int,
    order: int = 3,
    recency: int = 20
) -> Optional[float]:
    """
    Детекция бычьего паттерна Quasimodo (QML).
    
    Структура:
    1. Первый минимум (low1)
    2. Второй минимум ниже первого (low2 < low1) - sweep ликвидности
    3. Третий минимум выше второго (low3 > low2) - higher low
    
    Args:
        df: DataFrame со свечами
        idx: Текущий индекс для проверки
        order: Параметр для argrelextrema (количество свечей для подтверждения экстремума)
        recency: Максимальное количество свечей с момента формирования паттерна
    
    Returns:
        Цена sweep-минимума (low2) или None, если паттерн не найден
    """
    prices = df["low"][:idx + 1].values
    
    # Поиск локальных минимумов
    local_min_idx = argrelextrema(prices, np.less, order=order)[0]
    
    if len(local_min_idx) < 3:
        return None
    
    # Берём последние 3 минимума
    idx3 = local_min_idx[-1]  # Самый свежий
    idx2 = local_min_idx[-2]  # Sweep минимум
    idx1 = local_min_idx[-3]  # Первый минимум
    
    low1 = prices[idx1]
    low2 = prices[idx2]
    low3 = prices[idx3]
    
    # Проверка структуры QML:
    # low2 < low1 (sweep) AND low3 > low2 (higher low) AND свежий паттерн
    if low2 < low1 and low3 > low2 and (idx - idx3 < recency):
        return low2  # Возвращаем уровень sweep для SL
    
    return None


def detect_qml_bear(
    df: pd.DataFrame,
    idx: int,
    order: int = 3,
    recency: int = 20
) -> Optional[float]:
    """
    Детекция медвежьего паттерна Quasimodo (QML).
    
    Структура:
    1. Первый максимум (high1)
    2. Второй максимум выше первого (high2 > high1) - sweep ликвидности
    3. Третий максимум ниже второго (high3 < high2) - lower high
    
    Args:
        df: DataFrame со свечами
        idx: Текущий индекс для проверки
        order: Параметр для argrelextrema
        recency: Максимальное количество свечей с момента формирования
    
    Returns:
        Цена sweep-максимума (high2) или None, если паттерн не найден
    """
    prices = df["high"][:idx + 1].values
    
    # Поиск локальных максимумов
    local_max_idx = argrelextrema(prices, np.greater, order=order)[0]
    
    if len(local_max_idx) < 3:
        return None
    
    # Берём последние 3 максимума
    idx3 = local_max_idx[-1]  # Самый свежий
    idx2 = local_max_idx[-2]  # Sweep максимум
    idx1 = local_max_idx[-3]  # Первый максимум
    
    high1 = prices[idx1]
    high2 = prices[idx2]
    high3 = prices[idx3]
    
    # Проверка структуры QML:
    # high2 > high1 (sweep) AND high3 < high2 (lower high) AND свежий паттерн
    if high2 > high1 and high3 < high2 and (idx - idx3 < recency):
        return high2  # Возвращаем уровень sweep для SL
    
    return None


def find_swing_high(
    df: pd.DataFrame,
    lookback: int = 20,
    order: int = 3
) -> Optional[float]:
    """
    Поиск последнего swing high.
    
    Args:
        df: DataFrame со свечами
        lookback: Количество свечей для анализа
        order: Параметр для argrelextrema
    
    Returns:
        Цена swing high или None
    """
    if len(df) < lookback:
        return None
    
    prices = df["high"].tail(lookback).values
    local_max_idx = argrelextrema(prices, np.greater, order=order)[0]
    
    if len(local_max_idx) == 0:
        return None
    
    return float(prices[local_max_idx[-1]])


def find_swing_low(
    df: pd.DataFrame,
    lookback: int = 20,
    order: int = 3
) -> Optional[float]:
    """
    Поиск последнего swing low.
    
    Args:
        df: DataFrame со свечами
        lookback: Количество свечей для анализа
        order: Параметр для argrelextrema
    
    Returns:
        Цена swing low или None
    """
    if len(df) < lookback:
        return None
    
    prices = df["low"].tail(lookback).values
    local_min_idx = argrelextrema(prices, np.less, order=order)[0]
    
    if len(local_min_idx) == 0:
        return None
    
    return float(prices[local_min_idx[-1]])


def detect_higher_high(
    df: pd.DataFrame,
    lookback: int = 20,
    order: int = 3
) -> bool:
    """
    Проверка формирования Higher High (HH).
    
    Args:
        df: DataFrame со свечами
        lookback: Количество свечей для анализа
        order: Параметр для argrelextrema
    
    Returns:
        True если последний максимум выше предыдущего
    """
    prices = df["high"].tail(lookback).values
    local_max_idx = argrelextrema(prices, np.greater, order=order)[0]
    
    if len(local_max_idx) < 2:
        return False
    
    return prices[local_max_idx[-1]] > prices[local_max_idx[-2]]


def detect_lower_low(
    df: pd.DataFrame,
    lookback: int = 20,
    order: int = 3
) -> bool:
    """
    Проверка формирования Lower Low (LL).
    
    Args:
        df: DataFrame со свечами
        lookback: Количество свечей для анализа
        order: Параметр для argrelextrema
    
    Returns:
        True если последний минимум ниже предыдущего
    """
    prices = df["low"].tail(lookback).values
    local_min_idx = argrelextrema(prices, np.less, order=order)[0]
    
    if len(local_min_idx) < 2:
        return False
    
    return prices[local_min_idx[-1]] < prices[local_min_idx[-2]]


def detect_higher_low(
    df: pd.DataFrame,
    lookback: int = 20,
    order: int = 3
) -> bool:
    """
    Проверка формирования Higher Low (HL) - признак аптренда.
    
    Args:
        df: DataFrame со свечами
        lookback: Количество свечей для анализа
        order: Параметр для argrelextrema
    
    Returns:
        True если последний минимум выше предыдущего
    """
    prices = df["low"].tail(lookback).values
    local_min_idx = argrelextrema(prices, np.less, order=order)[0]
    
    if len(local_min_idx) < 2:
        return False
    
    return prices[local_min_idx[-1]] > prices[local_min_idx[-2]]


def detect_lower_high(
    df: pd.DataFrame,
    lookback: int = 20,
    order: int = 3
) -> bool:
    """
    Проверка формирования Lower High (LH) - признак даунтренда.
    
    Args:
        df: DataFrame со свечами
        lookback: Количество свечей для анализа
        order: Параметр для argrelextrema
    
    Returns:
        True если последний максимум ниже предыдущего
    """
    prices = df["high"].tail(lookback).values
    local_max_idx = argrelextrema(prices, np.greater, order=order)[0]
    
    if len(local_max_idx) < 2:
        return False
    
    return prices[local_max_idx[-1]] < prices[local_max_idx[-2]]
