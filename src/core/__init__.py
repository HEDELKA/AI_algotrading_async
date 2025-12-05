"""
Core modules: API client, indicators, patterns
"""

from .bingx_client import BingxClient
from .indicators import calculate_indicators
from .patterns import detect_qml_bull, detect_qml_bear
