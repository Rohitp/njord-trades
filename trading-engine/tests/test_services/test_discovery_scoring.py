"""
Tests for discovery scoring utilities.
"""

import pytest

from src.services.discovery.scoring import (
    liquidity_score,
    momentum_score,
    normalize_score,
    volatility_score,
)


class TestNormalizeScore:
    """Tests for normalize_score function."""

    def test_normalize_middle_value(self):
        """Test normalizing a middle value."""
        result = normalize_score(50.0, 0.0, 100.0)
        assert result == 0.5

    def test_normalize_min_value(self):
        """Test normalizing minimum value."""
        result = normalize_score(0.0, 0.0, 100.0)
        assert result == 0.0

    def test_normalize_max_value(self):
        """Test normalizing maximum value."""
        result = normalize_score(100.0, 0.0, 100.0)
        assert result == 1.0

    def test_normalize_below_min(self):
        """Test normalizing value below minimum (should clamp to 0.0)."""
        result = normalize_score(-10.0, 0.0, 100.0)
        assert result == 0.0

    def test_normalize_above_max(self):
        """Test normalizing value above maximum (should clamp to 1.0)."""
        result = normalize_score(150.0, 0.0, 100.0)
        assert result == 1.0

    def test_normalize_zero_range(self):
        """Test normalizing when min == max (should return 0.5)."""
        result = normalize_score(50.0, 50.0, 50.0)
        assert result == 0.5


class TestLiquidityScore:
    """Tests for liquidity_score function."""

    def test_liquidity_equal_volume(self):
        """Test liquidity score when volume equals average."""
        result = liquidity_score(1_000_000, 1_000_000)
        # 1.0x ratio normalized to [0, 3] range = 1.0/3.0 = 0.333...
        assert result == pytest.approx(0.333, abs=0.1)

    def test_liquidity_double_volume(self):
        """Test liquidity score when volume is double average."""
        result = liquidity_score(2_000_000, 1_000_000)
        assert result == pytest.approx(0.75, abs=0.1)  # 2.0x ratio = 0.75 score

    def test_liquidity_triple_volume(self):
        """Test liquidity score when volume is triple average."""
        result = liquidity_score(3_000_000, 1_000_000)
        assert result == pytest.approx(1.0, abs=0.1)  # 3.0x+ ratio = 1.0 score

    def test_liquidity_low_volume(self):
        """Test liquidity score when volume is below average."""
        result = liquidity_score(500_000, 1_000_000)
        assert result == pytest.approx(0.25, abs=0.1)  # 0.5x ratio = 0.25 score

    def test_liquidity_zero_average(self):
        """Test liquidity score when average volume is zero."""
        result = liquidity_score(1_000_000, 0.0)
        assert result == 0.0


class TestVolatilityScore:
    """Tests for volatility_score function."""

    def test_volatility_moderate(self):
        """Test volatility score for moderate volatility (sweet spot)."""
        result = volatility_score(0.3, min_vol=0.1, max_vol=0.5)
        # 0.3 is in [0.1, 0.5] range, normalized: (0.3-0.1)/(0.5-0.1) = 0.2/0.4 = 0.5
        assert result == pytest.approx(0.5, abs=0.01)

    def test_volatility_low(self):
        """Test volatility score for low volatility."""
        result = volatility_score(0.05, min_vol=0.1, max_vol=0.5)
        # 0.05 < 0.1, normalized to [0, 0.1] = 0.05/0.1 = 0.5
        assert result == pytest.approx(0.5, abs=0.01)

    def test_volatility_high(self):
        """Test volatility score for high volatility."""
        result = volatility_score(0.8, min_vol=0.1, max_vol=0.5)
        assert 0.0 <= result < 0.5  # Should be penalized

    def test_volatility_at_min(self):
        """Test volatility score at minimum threshold."""
        result = volatility_score(0.1, min_vol=0.1, max_vol=0.5)
        assert result == 0.0

    def test_volatility_at_max(self):
        """Test volatility score at maximum threshold."""
        result = volatility_score(0.5, min_vol=0.1, max_vol=0.5)
        assert result == 1.0


class TestMomentumScore:
    """Tests for momentum_score function."""

    def test_momentum_positive_moderate(self):
        """Test momentum score for moderate positive momentum (sweet spot)."""
        result = momentum_score(0.05, min_change=0.02, max_change=0.10)
        # 0.05 is in [0.02, 0.10] range, normalized: (0.05-0.02)/(0.10-0.02) = 0.03/0.08 = 0.375
        assert result == pytest.approx(0.375, abs=0.01)

    def test_momentum_negative(self):
        """Test momentum score for negative momentum."""
        result = momentum_score(-0.05, min_change=0.02, max_change=0.10)
        assert 0.0 <= result < 0.5  # Should be penalized

    def test_momentum_extreme_positive(self):
        """Test momentum score for extreme positive momentum (overbought)."""
        result = momentum_score(0.15, min_change=0.02, max_change=0.10)
        # 0.15 > 0.10, inverted: 1.0 - normalize(0.15, 0.10, 0.20) = 1.0 - 0.5 = 0.5
        assert result == pytest.approx(0.5, abs=0.01)

    def test_momentum_at_min(self):
        """Test momentum score at minimum threshold."""
        result = momentum_score(0.02, min_change=0.02, max_change=0.10)
        assert result == 0.0

    def test_momentum_at_max(self):
        """Test momentum score at maximum threshold."""
        result = momentum_score(0.10, min_change=0.02, max_change=0.10)
        assert result == 1.0

    def test_momentum_zero(self):
        """Test momentum score for zero change."""
        result = momentum_score(0.0, min_change=0.02, max_change=0.10)
        # 0.0 < 0.02, normalized to [-0.10, 0.02]: (0.0 - (-0.10))/(0.02 - (-0.10)) = 0.1/0.12 = 0.833
        assert result == pytest.approx(0.833, abs=0.01)

