"""
Асинхронный real-time торговый бот.
"""

import asyncio
import signal
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from dataclasses import dataclass

from ..core.bingx_client import BingxClient
from ..core.indicators import calculate_indicators
from ..strategies.base import StrategyConfig
from ..strategies.macd_ema_bb import MacdEmaBbStrategy

logger = logging.getLogger(__name__)


@dataclass
class BotConfig:
    """Конфигурация бота."""
    # API
    api_key: str
    api_secret: str
    
    # Trading
    symbol: str = "LTCUSDT"
    interval: str = "1h"
    leverage: int = 20
    risk_percent: float = 0.01
    
    # Mode
    use_testnet: bool = True
    
    # Strategy
    strategy_config: Optional[StrategyConfig] = None


class TradingBot:
    """
    Асинхронный торговый бот для BingX.
    
    Особенности:
    - Проверка сигналов каждый час (на закрытии свечи)
    - Автоматическое открытие/закрытие позиций
    - Установка SL/TP ордеров
    - Graceful shutdown по SIGINT/SIGTERM
    """
    
    def __init__(self, config: BotConfig):
        """
        Инициализация бота.
        
        Args:
            config: Конфигурация бота
        """
        self.config = config
        self.client = BingxClient(
            api_key=config.api_key,
            api_secret=config.api_secret,
            symbol=config.symbol,
            use_testnet=config.use_testnet
        )
        
        strategy_config = config.strategy_config or StrategyConfig()
        self.strategy = MacdEmaBbStrategy(strategy_config)
        
        self._running = False
        self._position: Optional[Dict[str, Any]] = None
    
    async def start(self):
        """Запуск бота."""
        logger.info("=" * 50)
        logger.info("🚀 Запуск торгового бота")
        logger.info(f"   Symbol: {self.config.symbol}")
        logger.info(f"   Interval: {self.config.interval}")
        logger.info(f"   Leverage: {self.config.leverage}x")
        logger.info(f"   Risk: {self.config.risk_percent * 100:.1f}%")
        logger.info(f"   Mode: {'TESTNET' if self.config.use_testnet else 'MAINNET'}")
        logger.info("=" * 50)
        
        self._running = True
        
        # Настройка обработчиков сигналов
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
        
        try:
            # Синхронизация времени
            await self.client.get_server_time()
            logger.info("✅ Время синхронизировано с сервером")
            
            # Установка плеча
            await self._set_leverage()
            
            # Основной цикл
            await self._main_loop()
        
        except Exception as e:
            logger.error(f"❌ Критическая ошибка: {e}")
            raise
        
        finally:
            await self.client.close()
            logger.info("🛑 Бот остановлен")
    
    async def stop(self):
        """Graceful shutdown бота."""
        logger.info("⏳ Получен сигнал остановки...")
        self._running = False
    
    async def _set_leverage(self):
        """Установка кредитного плеча."""
        try:
            await self.client.set_leverage(self.config.leverage, "LONG")
            await self.client.set_leverage(self.config.leverage, "SHORT")
            logger.info(f"✅ Плечо установлено: {self.config.leverage}x")
        except Exception as e:
            logger.warning(f"⚠️  Ошибка установки плеча: {e}")
    
    async def _main_loop(self):
        """Основной цикл бота."""
        while self._running:
            now = datetime.now(timezone.utc)
            
            # Проверка на начало нового часа (для интервала 1h)
            if self._should_check_signal(now):
                logger.info(f"\n{'─' * 40}")
                logger.info(f"🔍 Проверка сигналов [{now.strftime('%Y-%m-%d %H:%M')} UTC]")
                
                try:
                    await self._check_and_trade()
                except Exception as e:
                    logger.error(f"❌ Ошибка при проверке: {e}")
                
                # Пауза чтобы не проверять повторно в ту же минуту
                await asyncio.sleep(60)
            else:
                # Проверка каждую минуту
                await asyncio.sleep(30)
    
    def _should_check_signal(self, now: datetime) -> bool:
        """Определение, нужно ли проверять сигнал."""
        interval = self.config.interval
        
        if interval == "1h":
            return now.minute == 0 and now.second < 30
        elif interval == "4h":
            return now.hour % 4 == 0 and now.minute == 0 and now.second < 30
        elif interval == "15m":
            return now.minute % 15 == 0 and now.second < 30
        elif interval == "5m":
            return now.minute % 5 == 0 and now.second < 30
        
        # По умолчанию проверяем каждый час
        return now.minute == 0 and now.second < 30
    
    async def _check_and_trade(self):
        """Проверка сигналов и торговля."""
        # Получение свечей
        df = await self.client.get_klines(interval=self.config.interval, limit=200)
        
        # Расчёт индикаторов
        config = self.strategy.config
        df = calculate_indicators(
            df,
            ema_period=config.ema_period,
            bb_std=config.bb_std,
            atr_period=config.atr_period,
            vol_period=config.vol_period
        )
        
        idx = len(df) - 1
        
        # Проверка открытых позиций
        in_position = await self.client.has_open_position()
        
        if in_position:
            logger.info("📊 Есть открытая позиция")
            # Здесь можно добавить логику trailing stop
            return
        
        # Проверка сигнала
        signal = self.strategy.generate_signal(df, idx)
        
        if signal is None:
            logger.info("❌ Нет сигнала")
            return
        
        logger.info(f"✨ Сигнал: {signal.side.upper()}")
        
        # Получение текущей цены и баланса
        current_price = await self.client.get_mark_price()
        if current_price is None:
            logger.error("Не удалось получить цену")
            return
        
        balance = await self.client.get_available_balance()
        if balance <= 0:
            logger.error("Недостаточно баланса")
            return
        
        logger.info(f"   Price: {current_price}")
        logger.info(f"   Balance: {balance:.2f} USDT")
        
        # Расчёт SL/TP по текущей цене
        atr = df["atr"].iloc[idx]
        sl, tp = self.strategy.calculate_sl_tp(current_price, signal.side, atr)
        
        # Расчёт размера позиции
        quantity = self.client.calculate_quantity(
            balance=balance,
            entry_price=current_price,
            stop_loss=sl,
            risk_percent=self.config.risk_percent
        )
        
        if quantity <= 0:
            logger.error("Расчётное количество = 0")
            return
        
        logger.info(f"   SL: {sl:.4f}")
        logger.info(f"   TP: {tp:.4f}")
        logger.info(f"   Qty: {quantity}")
        
        # Открытие позиции
        try:
            order_resp = await self.client.place_market_order(
                side=signal.side,
                quantity=quantity,
                stop_loss=sl,
                take_profit=tp
            )
            
            if order_resp.get("code") == 0:
                entry_price = float(order_resp["data"].get("avgPrice", current_price))
                
                logger.info(f"✅ Позиция открыта!")
                logger.info(f"   Entry: {entry_price}")
                
                # Сохранение информации о позиции
                self._position = {
                    "side": signal.side,
                    "entry": entry_price,
                    "sl": sl,
                    "tp": tp,
                    "qty": quantity
                }
            else:
                logger.error(f"❌ Ошибка открытия: {order_resp}")
        
        except Exception as e:
            logger.error(f"❌ Исключение при открытии: {e}")


async def run_bot(config: BotConfig):
    """
    Запуск бота.
    
    Args:
        config: Конфигурация бота
    """
    bot = TradingBot(config)
    await bot.start()
