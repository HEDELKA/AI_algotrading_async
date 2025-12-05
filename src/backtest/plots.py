"""
Визуализация результатов бэктеста.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .engine import BacktestResult, Trade


def plot_equity_curve(
    result: BacktestResult,
    df: Optional[pd.DataFrame] = None,
    title: str = "Equity Curve",
    save_path: Optional[str] = None
) -> None:
    """
    Построение графика equity curve.
    
    Args:
        result: Результаты бэктеста
        df: DataFrame со свечами (для timestamps)
        title: Заголовок графика
        save_path: Путь для сохранения (если указан)
    """
    if not result.trades:
        print("Нет сделок для построения equity curve.")
        return
    
    equity = result.equity_curve
    
    # Создание timestamps для оси X
    if df is not None and "timestamp" in df.columns:
        # Используем timestamps из df для каждой сделки
        timestamps = [df["timestamp"].iloc[result.trades[0].entry_idx]]
        for trade in result.trades:
            timestamps.append(df["timestamp"].iloc[trade.exit_idx])
    else:
        # Просто индексы
        timestamps = list(range(len(equity)))
    
    # График
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(timestamps, equity, linewidth=2, color="#2ecc71")
    ax.fill_between(timestamps, 1.0, equity, alpha=0.3, color="#2ecc71")
    
    ax.axhline(y=1.0, color="#e74c3c", linestyle="--", alpha=0.5, label="Начальный капитал")
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Дата" if df is not None else "Сделка №")
    ax.set_ylabel("Капитал (x)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Аннотации
    final_equity = equity[-1]
    total_pnl = (final_equity - 1) * 100
    ax.annotate(
        f"Итог: {final_equity:.2f}x ({total_pnl:+.2f}%)",
        xy=(timestamps[-1], final_equity),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"График сохранён: {save_path}")
    else:
        plt.show()


def plot_drawdown(
    result: BacktestResult,
    title: str = "Drawdown",
    save_path: Optional[str] = None
) -> None:
    """
    Построение графика просадки.
    
    Args:
        result: Результаты бэктеста
        title: Заголовок графика
        save_path: Путь для сохранения
    """
    equity = np.array(result.equity_curve)
    running_max = np.maximum.accumulate(equity)
    drawdown = (running_max - equity) / running_max * 100
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    ax.fill_between(range(len(drawdown)), 0, -drawdown, alpha=0.5, color="#e74c3c")
    ax.plot(range(len(drawdown)), -drawdown, linewidth=1, color="#c0392b")
    
    ax.axhline(y=-result.max_drawdown, color="#8e44ad", linestyle="--", 
               label=f"Max DD: {result.max_drawdown:.2f}%")
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Сделка №")
    ax.set_ylabel("Просадка (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


def plot_trades_distribution(
    result: BacktestResult,
    title: str = "Распределение PNL по сделкам",
    save_path: Optional[str] = None
) -> None:
    """
    Гистограмма распределения PNL по сделкам.
    
    Args:
        result: Результаты бэктеста
        title: Заголовок
        save_path: Путь для сохранения
    """
    if not result.trades:
        print("Нет сделок для построения гистограммы.")
        return
    
    pnls = [t.pnl_pct for t in result.trades]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = ["#2ecc71" if p > 0 else "#e74c3c" for p in pnls]
    ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
    
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axhline(y=result.avg_pnl, color="#3498db", linestyle="--",
               label=f"Средний PNL: {result.avg_pnl:.2f}%")
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Сделка №")
    ax.set_ylabel("PNL (%)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


def plot_combined_report(
    result: BacktestResult,
    df: Optional[pd.DataFrame] = None,
    title: str = "Backtest Report",
    save_path: Optional[str] = None
) -> None:
    """
    Комплексный отчёт с несколькими графиками.
    
    Args:
        result: Результаты бэктеста
        df: DataFrame со свечами
        title: Заголовок
        save_path: Путь для сохранения
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Equity Curve
    ax1 = fig.add_subplot(2, 2, 1)
    equity = result.equity_curve
    ax1.plot(equity, linewidth=2, color="#2ecc71")
    ax1.fill_between(range(len(equity)), 1.0, equity, alpha=0.3, color="#2ecc71")
    ax1.axhline(y=1.0, color="#e74c3c", linestyle="--", alpha=0.5)
    ax1.set_title("Equity Curve", fontweight="bold")
    ax1.set_ylabel("Капитал (x)")
    ax1.grid(True, alpha=0.3)
    
    # Drawdown
    ax2 = fig.add_subplot(2, 2, 2)
    equity_arr = np.array(result.equity_curve)
    running_max = np.maximum.accumulate(equity_arr)
    drawdown = (running_max - equity_arr) / running_max * 100
    ax2.fill_between(range(len(drawdown)), 0, -drawdown, alpha=0.5, color="#e74c3c")
    ax2.axhline(y=-result.max_drawdown, color="#8e44ad", linestyle="--")
    ax2.set_title(f"Drawdown (Max: {result.max_drawdown:.2f}%)", fontweight="bold")
    ax2.set_ylabel("Просадка (%)")
    ax2.grid(True, alpha=0.3)
    
    # PNL Distribution
    ax3 = fig.add_subplot(2, 2, 3)
    pnls = [t.pnl_pct for t in result.trades]
    colors = ["#2ecc71" if p > 0 else "#e74c3c" for p in pnls]
    ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
    ax3.axhline(y=0, color="black", linewidth=0.5)
    ax3.axhline(y=result.avg_pnl, color="#3498db", linestyle="--")
    ax3.set_title("Распределение PNL", fontweight="bold")
    ax3.set_xlabel("Сделка №")
    ax3.set_ylabel("PNL (%)")
    ax3.grid(True, alpha=0.3, axis="y")
    
    # Statistics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    
    stats_text = f"""
    СТАТИСТИКА БЭКТЕСТА
    ───────────────────────
    
    📈 Total PNL:     {result.total_pnl_pct:+.2f}%
    🎯 Win Rate:      {result.win_rate:.2f}%
    📊 Всего сделок:  {result.num_trades}
    📉 Avg PNL:       {result.avg_pnl:+.2f}%
    ⚠️  Max Drawdown: {result.max_drawdown:.2f}%
    
    ───────────────────────
    Wins:  {sum(1 for t in result.trades if t.pnl_pct > 0)}
    Losses: {sum(1 for t in result.trades if t.pnl_pct <= 0)}
    """
    
    ax4.text(0.5, 0.5, stats_text, transform=ax4.transAxes,
             fontsize=12, verticalalignment="center", horizontalalignment="center",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray"))
    
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Отчёт сохранён: {save_path}")
    else:
        plt.show()
