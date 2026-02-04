"""
Base protocol and data structures for symbol pickers.

Defines the interface that all pickers (Metric, Fuzzy, LLM) must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class PickerResult:
    """
    Result from a symbol picker.

    Each picker returns a list of symbols with scores/rankings.
    """

    symbol: str
    score: float  # 0.0-1.0, higher is better
    reason: str  # Why this symbol was picked
    metadata: dict | None = None  # Picker-specific metadata


class SymbolPicker(ABC):
    """
    Abstract base class for symbol pickers.

    Each picker implements a different strategy for discovering trading symbols:
    - MetricPicker: Pure quantitative filters
    - FuzzyPicker: Weighted multi-factor scoring
    - LLMPicker: LLM-powered context-aware selection
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Picker name for logging and identification."""
        pass

    @abstractmethod
    async def pick(self, context: dict | None = None) -> List[PickerResult]:
        """
        Pick symbols based on picker's strategy.

        Args:
            context: Optional context (portfolio state, market conditions, etc.)

        Returns:
            List of PickerResult objects, sorted by score (highest first)
        """
        pass

