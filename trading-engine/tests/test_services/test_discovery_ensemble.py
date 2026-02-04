"""
Tests for ensemble combiner.
"""

import pytest

from src.services.discovery.ensemble import EnsembleCombiner
from src.services.discovery.pickers.base import PickerResult


class TestEnsembleCombiner:
    """Tests for EnsembleCombiner."""

    def test_combine_single_picker(self):
        """Test combining results from a single picker."""
        combiner = EnsembleCombiner()
        
        metric_results = [
            PickerResult(symbol="AAPL", score=1.0, reason="Passed filters"),
            PickerResult(symbol="MSFT", score=1.0, reason="Passed filters"),
        ]
        
        results = combiner.combine(metric_results=metric_results)
        
        assert len(results) == 2
        assert results[0].symbol == "AAPL"
        assert results[0].score == 1.0
        assert "metric" in results[0].reason.lower()

    def test_combine_multiple_pickers_same_symbol(self):
        """Test that symbols appearing in multiple pickers are deduplicated."""
        combiner = EnsembleCombiner(
            metric_weight=0.3,
            fuzzy_weight=0.4,
            llm_weight=0.3,
        )
        
        # Same symbol from different pickers
        metric_results = [
            PickerResult(symbol="AAPL", score=1.0, reason="Passed filters"),
        ]
        fuzzy_results = [
            PickerResult(symbol="AAPL", score=0.8, reason="Good liquidity"),
        ]
        llm_results = [
            PickerResult(symbol="AAPL", score=0.9, reason="Strong momentum"),
        ]
        
        results = combiner.combine(
            metric_results=metric_results,
            fuzzy_results=fuzzy_results,
            llm_results=llm_results,
        )
        
        # Should have only one AAPL result
        assert len(results) == 1
        assert results[0].symbol == "AAPL"
        
        # Composite score should be weighted average
        expected_score = (1.0 * 0.3 + 0.8 * 0.4 + 0.9 * 0.3) / (0.3 + 0.4 + 0.3)
        assert abs(results[0].score - expected_score) < 0.001
        
        # Should include all pickers in metadata
        assert "pickers" in results[0].metadata
        assert set(results[0].metadata["pickers"]) == {"metric", "fuzzy", "llm"}

    def test_combine_different_symbols(self):
        """Test combining results with different symbols from each picker."""
        combiner = EnsembleCombiner()
        
        metric_results = [
            PickerResult(symbol="AAPL", score=1.0, reason="Passed filters"),
        ]
        fuzzy_results = [
            PickerResult(symbol="MSFT", score=0.7, reason="Good score"),
        ]
        llm_results = [
            PickerResult(symbol="GOOGL", score=0.8, reason="LLM recommendation"),
        ]
        
        results = combiner.combine(
            metric_results=metric_results,
            fuzzy_results=fuzzy_results,
            llm_results=llm_results,
        )
        
        # Should have all three symbols
        assert len(results) == 3
        
        # Should be sorted by score (highest first)
        assert results[0].symbol == "AAPL"  # score 1.0
        assert results[1].symbol == "GOOGL"  # score 0.8
        assert results[2].symbol == "MSFT"  # score 0.7

    def test_combine_weighted_average(self):
        """Test that weighted average is calculated correctly."""
        combiner = EnsembleCombiner(
            metric_weight=0.5,
            fuzzy_weight=0.3,
            llm_weight=0.2,
        )
        
        # Same symbol, different scores
        metric_results = [
            PickerResult(symbol="AAPL", score=1.0, reason="Metric"),
        ]
        fuzzy_results = [
            PickerResult(symbol="AAPL", score=0.5, reason="Fuzzy"),
        ]
        llm_results = [
            PickerResult(symbol="AAPL", score=0.0, reason="LLM"),
        ]
        
        results = combiner.combine(
            metric_results=metric_results,
            fuzzy_results=fuzzy_results,
            llm_results=llm_results,
        )
        
        # Weighted average: (1.0*0.5 + 0.5*0.3 + 0.0*0.2) / (0.5+0.3+0.2) = 0.65
        expected_score = (1.0 * 0.5 + 0.5 * 0.3 + 0.0 * 0.2) / 1.0
        assert abs(results[0].score - expected_score) < 0.001

    def test_combine_empty_results(self):
        """Test that empty results return empty list."""
        combiner = EnsembleCombiner()
        results = combiner.combine()
        assert results == []

    def test_combine_reason_combination(self):
        """Test that reasons from multiple pickers are combined."""
        combiner = EnsembleCombiner()
        
        metric_results = [
            PickerResult(symbol="AAPL", score=1.0, reason="Passed volume filter"),
        ]
        fuzzy_results = [
            PickerResult(symbol="AAPL", score=0.8, reason="Good liquidity score"),
        ]
        
        results = combiner.combine(
            metric_results=metric_results,
            fuzzy_results=fuzzy_results,
        )
        
        assert len(results) == 1
        assert "metric" in results[0].reason.lower()
        assert "fuzzy" in results[0].reason.lower()
        assert "|" in results[0].reason  # Separator

    def test_combine_from_dict(self):
        """Test combining from dictionary of results."""
        combiner = EnsembleCombiner()
        
        picker_results = {
            "metric": [
                PickerResult(symbol="AAPL", score=1.0, reason="Metric"),
            ],
            "fuzzy": [
                PickerResult(symbol="MSFT", score=0.7, reason="Fuzzy"),
            ],
            "llm": [
                PickerResult(symbol="GOOGL", score=0.8, reason="LLM"),
            ],
        }
        
        results = combiner.combine_from_dict(picker_results)
        
        assert len(results) == 3
        symbols = {r.symbol for r in results}
        assert symbols == {"AAPL", "MSFT", "GOOGL"}

    def test_weight_normalization(self):
        """Test that weights are normalized correctly."""
        # Weights that don't sum to 1.0 should be normalized
        combiner = EnsembleCombiner(
            metric_weight=0.6,
            fuzzy_weight=0.4,
            llm_weight=0.0,
        )
        
        # Should normalize to sum to 1.0
        total = combiner.metric_weight + combiner.fuzzy_weight + combiner.llm_weight
        assert abs(total - 1.0) < 0.001

    def test_case_insensitive_symbols(self):
        """Test that symbol case is normalized."""
        combiner = EnsembleCombiner()
        
        metric_results = [
            PickerResult(symbol="AAPL", score=1.0, reason="Metric"),
        ]
        fuzzy_results = [
            PickerResult(symbol="aapl", score=0.8, reason="Fuzzy"),  # Lowercase
        ]
        
        results = combiner.combine(
            metric_results=metric_results,
            fuzzy_results=fuzzy_results,
        )
        
        # Should deduplicate (same symbol, different case)
        assert len(results) == 1
        assert results[0].symbol == "AAPL"  # Uppercase

