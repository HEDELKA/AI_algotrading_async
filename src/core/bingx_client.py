"""
Асинхронный клиент для BingX Perpetual Futures API.
Основан на документации: https://bingx-api.github.io/docs/
"""

import time
import hmac
import hashlib
import json
import logging
from typing import Optional, Dict, Any, List

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)


class BingxClient:
    """Асинхронный клиент для BingX Perpetual Futures API."""
    
    # Mainnet
    BASE_URL = "https://open-api.bingx.com"
    # Testnet (VST = Virtual Standard Token)
    TESTNET_URL = "https://open-api-vst.bingx.com"
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbol: str = "LTCUSDT",
        use_testnet: bool = True
    ):
        """
        Инициализация клиента.
        
        Args:
            api_key: API ключ BingX
            api_secret: API секрет BingX
            symbol: Торговая пара (например, LTCUSDT)
            use_testnet: Использовать тестовую сеть
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = self._to_bingx_symbol(symbol)
        self.base_url = self.TESTNET_URL if use_testnet else self.BASE_URL
        self.time_offset = 0
        self._session: Optional[aiohttp.ClientSession] = None
    
    @staticmethod
    def _to_bingx_symbol(symbol: str) -> str:
        """Конвертация символа в формат BingX (BTC-USDT)."""
        if "-" in symbol:
            return symbol
        # BTCUSDT -> BTC-USDT
        if symbol.endswith("USDT"):
            return symbol[:-4] + "-USDT"
        return symbol
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Получение или создание HTTP сессии."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Закрытие HTTP сессии."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _sign(self, query: str) -> str:
        """Создание HMAC-SHA256 подписи."""
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
    
    def _build_params(self, params: Dict[str, Any]) -> str:
        """Построение строки параметров с timestamp."""
        params = {k: v for k, v in params.items() if v is not None}
        sorted_keys = sorted(params.keys())
        query = "&".join([f"{k}={params[k]}" for k in sorted_keys])
        timestamp = int(time.time() * 1000) + self.time_offset
        if query:
            return f"{query}&timestamp={timestamp}"
        return f"timestamp={timestamp}"
    
    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = True
    ) -> Dict[str, Any]:
        """
        Выполнение HTTP запроса к API.
        
        Args:
            method: HTTP метод (GET, POST, DELETE)
            path: API endpoint
            params: Параметры запроса
            signed: Требуется ли подпись
        
        Returns:
            JSON ответ от API
        """
        session = await self._get_session()
        params = params or {}
        
        if signed:
            query = self._build_params(params)
            signature = self._sign(query)
            url = f"{self.base_url}{path}?{query}&signature={signature}"
            headers = {"X-BX-APIKEY": self.api_key}
        else:
            url = f"{self.base_url}{path}"
            if params:
                url += "?" + "&".join([f"{k}={v}" for k, v in params.items()])
            headers = {}
        
        try:
            async with session.request(method, url, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                
                if data.get("code") != 0:
                    logger.error(f"API Error: {data}")
                
                return data
        except aiohttp.ClientError as e:
            logger.error(f"Request error: {e}")
            raise
    
    async def _public_request(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Публичный запрос (без подписи)."""
        return await self._request("GET", path, params, signed=False)
    
    # ==================== Market Data ====================
    
    async def get_server_time(self) -> int:
        """Получение времени сервера и синхронизация."""
        path = "/openApi/swap/v2/server/time"
        data = await self._public_request(path)
        
        if data.get("code") == 0:
            server_time = int(data["data"]["serverTime"])
            local_time = int(time.time() * 1000)
            self.time_offset = server_time - local_time
            return server_time
        return 0
    
    async def get_klines(
        self,
        symbol: Optional[str] = None,
        interval: str = "1h",
        limit: int = 200,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Получение свечей (OHLCV данных).
        
        Args:
            symbol: Торговая пара
            interval: Таймфрейм (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            limit: Количество свечей (max 1000)
            start_time: Начальное время (ms)
            end_time: Конечное время (ms)
        
        Returns:
            DataFrame с колонками: timestamp, open, high, low, close, volume
        """
        path = "/openApi/swap/v2/quote/klines"
        params = {
            "symbol": symbol or self.symbol,
            "interval": interval,
            "limit": min(limit, 1000)
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        data = await self._public_request(path, params)
        
        if data.get("code") != 0:
            raise Exception(f"Klines error: {data.get('msg')}")
        
        klines = data["data"]
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df[["open", "high", "low", "close", "volume"]] = df[[
            "open", "high", "low", "close", "volume"
        ]].astype(float)
        
        return df
    
    async def get_mark_price(self, symbol: Optional[str] = None) -> Optional[float]:
        """Получение mark price (цены маркировки)."""
        path = "/openApi/swap/v2/quote/premiumIndex"
        params = {"symbol": symbol or self.symbol}
        
        try:
            data = await self._public_request(path, params)
            if data.get("code") == 0 and "data" in data:
                if isinstance(data["data"], list) and data["data"]:
                    mark_price = data["data"][0].get("markPrice")
                elif isinstance(data["data"], dict):
                    mark_price = data["data"].get("markPrice")
                else:
                    return None
                return float(mark_price) if mark_price else None
        except Exception as e:
            logger.error(f"Get mark price error: {e}")
        return None
    
    # ==================== Account ====================
    
    async def get_balance(self) -> Dict[str, Any]:
        """Получение баланса аккаунта."""
        path = "/openApi/swap/v3/user/balance"
        return await self._request("GET", path)
    
    async def get_available_balance(self) -> float:
        """Получение доступного баланса в USDT."""
        data = await self.get_balance()
        if data.get("code") == 0 and "data" in data:
            balance_list = data["data"]
            if isinstance(balance_list, list) and balance_list:
                return float(balance_list[0].get("availableBalance", 0))
        return 0.0
    
    async def get_positions(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Получение открытых позиций."""
        path = "/openApi/swap/v2/user/positions"
        
        sym = symbol or self.symbol
        # Конвертация формата символа
        if "-" not in sym:
            sym = sym[:-4] + "-" + sym[-4:]
        
        params = {
            "symbol": sym,
            "timestamp": int(time.time() * 1000)
        }
        
        return await self._request("GET", path, params)
    
    async def has_open_position(self, symbol: Optional[str] = None) -> bool:
        """Проверка наличия открытой позиции."""
        data = await self.get_positions(symbol)
        if data.get("code") == 0 and "data" in data and data["data"]:
            for pos in data["data"]:
                if float(pos.get("positionAmt", 0)) != 0:
                    return True
        return False
    
    # ==================== Trading ====================
    
    async def set_leverage(
        self,
        leverage: int,
        side: str = "BOTH",
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Установка кредитного плеча.
        
        Args:
            leverage: Размер плеча (1-125)
            side: LONG, SHORT или BOTH
            symbol: Торговая пара
        """
        path = "/openApi/swap/v2/trade/leverage"
        params = {
            "symbol": symbol or self.symbol,
            "side": side,
            "leverage": leverage
        }
        return await self._request("POST", path, params)
    
    async def place_market_order(
        self,
        side: str,
        quantity: float,
        symbol: Optional[str] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Размещение рыночного ордера.
        
        Args:
            side: 'long' или 'short'
            quantity: Размер позиции
            symbol: Торговая пара
            stop_loss: Цена стоп-лосса
            take_profit: Цена тейк-профита
        
        Returns:
            Ответ API с деталями ордера
        """
        path = "/openApi/swap/v2/trade/order"
        
        side_param = "BUY" if side.lower() == "long" else "SELL"
        position_side = "LONG" if side.lower() == "long" else "SHORT"
        
        params = {
            "symbol": symbol or self.symbol,
            "side": side_param,
            "positionSide": position_side,
            "type": "MARKET",
            "quantity": quantity,
            "recvWindow": 5000,
            "timeInForce": "GTC"
        }
        
        # Добавление стоп-лосса
        if stop_loss is not None:
            sl_param = {
                "type": "STOP_MARKET",
                "stopPrice": stop_loss,
                "price": stop_loss,
                "workingType": "MARK_PRICE"
            }
            params["stopLoss"] = json.dumps(sl_param)
        
        # Добавление тейк-профита
        if take_profit is not None:
            tp_param = {
                "type": "TAKE_PROFIT_MARKET",
                "stopPrice": take_profit,
                "price": take_profit,
                "workingType": "MARK_PRICE"
            }
            params["takeProfit"] = json.dumps(tp_param)
        
        return await self._request("POST", path, params)
    
    async def set_stop_loss(
        self,
        stop_price: float,
        quantity: float,
        side: str,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Установка стоп-лосса для позиции.
        
        Args:
            stop_price: Цена стопа
            quantity: Размер позиции
            side: 'long' или 'short' (позиция, которую закрываем)
        """
        path = "/openApi/swap/v2/trade/order"
        
        # Для закрытия long нужен SELL, для short - BUY
        order_side = "SELL" if side.lower() == "long" else "BUY"
        position_side = "LONG" if side.lower() == "long" else "SHORT"
        
        params = {
            "symbol": symbol or self.symbol,
            "side": order_side,
            "positionSide": position_side,
            "type": "STOP_MARKET",
            "stopPrice": stop_price,
            "price": stop_price,
            "quantity": quantity,
            "workingType": "MARK_PRICE",
            "recvWindow": 5000
        }
        
        return await self._request("POST", path, params)
    
    async def set_take_profit(
        self,
        tp_price: float,
        quantity: float,
        side: str,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Установка тейк-профита для позиции.
        
        Args:
            tp_price: Цена тейка
            quantity: Размер позиции
            side: 'long' или 'short'
        """
        path = "/openApi/swap/v2/trade/order"
        
        order_side = "SELL" if side.lower() == "long" else "BUY"
        position_side = "LONG" if side.lower() == "long" else "SHORT"
        
        params = {
            "symbol": symbol or self.symbol,
            "side": order_side,
            "positionSide": position_side,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": tp_price,
            "quantity": quantity,
            "workingType": "MARK_PRICE",
            "recvWindow": 5000
        }
        
        return await self._request("POST", path, params)
    
    async def set_multiple_sl(
        self,
        sl_levels: List[float],
        quantity: float,
        side: str,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Установка нескольких уровней стоп-лосса.
        
        Args:
            sl_levels: Список цен стопов
            quantity: Общий размер позиции
            side: 'long' или 'short'
        """
        qty_per_level = quantity / len(sl_levels)
        results = []
        
        for sl_price in sl_levels:
            result = await self.set_stop_loss(sl_price, qty_per_level, side, symbol)
            results.append(result)
            logger.info(f"SL set at {sl_price}")
        
        return results
    
    async def set_multiple_tp(
        self,
        tp_levels: List[float],
        quantity: float,
        side: str,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Установка нескольких уровней тейк-профита.
        
        Args:
            tp_levels: Список цен тейков
            quantity: Общий размер позиции
            side: 'long' или 'short'
        """
        qty_per_level = quantity / len(tp_levels)
        results = []
        
        for tp_price in tp_levels:
            result = await self.set_take_profit(tp_price, qty_per_level, side, symbol)
            results.append(result)
            logger.info(f"TP set at {tp_price}")
        
        return results
    
    # ==================== Order Management ====================
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Получение открытых ордеров."""
        path = "/openApi/swap/v2/trade/allOpenOrders"
        params = {"symbol": symbol or self.symbol}
        return await self._request("GET", path, params)
    
    async def cancel_order(
        self,
        order_id: str,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """Отмена ордера по ID."""
        path = "/openApi/swap/v2/trade/cancel"
        params = {
            "symbol": symbol or self.symbol,
            "orderId": order_id
        }
        return await self._request("POST", path, params)
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Отмена всех открытых ордеров."""
        path = "/openApi/swap/v2/trade/allOpenOrders"
        params = {"symbol": symbol or self.symbol}
        return await self._request("DELETE", path, params)
    
    # ==================== Helpers ====================
    
    @staticmethod
    def count_decimal_places(number: float) -> int:
        """Подсчёт десятичных знаков числа."""
        s = str(number).rstrip("0")
        if "." in s:
            return len(s.split(".")[1])
        return 0
    
    def calculate_quantity(
        self,
        balance: float,
        entry_price: float,
        stop_loss: float,
        risk_percent: float = 0.01
    ) -> float:
        """
        Расчёт размера позиции на основе риска.
        
        Args:
            balance: Доступный баланс
            entry_price: Цена входа
            stop_loss: Цена стоп-лосса
            risk_percent: Процент риска (0.01 = 1%)
        
        Returns:
            Размер позиции в базовой валюте
        """
        risk_usdt = balance * risk_percent
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            return 0.0
        
        quantity = risk_usdt / risk_per_unit
        
        # Округление на основе точности цены
        precision = self.count_decimal_places(entry_price)
        if precision >= 3:
            qty_precision = 0
        elif precision >= 2:
            qty_precision = 1
        elif precision >= 1:
            qty_precision = 2
        else:
            qty_precision = 3
        
        return round(quantity, qty_precision)
