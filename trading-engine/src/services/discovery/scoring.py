"""
Shared scoring utilities for symbol discovery.

Provides common scoring functions used by multiple pickers.
"""


def normalize_score(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize a value to 0.0-1.0 range.

    Args:
        value: Value to normalize
        min_val: Minimum expected value
        max_val: Maximum expected value

    Returns:
        Normalized score between 0.0 and 1.0 (clamped)
    """
    if max_val == min_val:
        return 0.5  # Avoid division by zero
    
    normalized = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))  # Clamp to [0, 1]


def liquidity_score(volume: float, avg_volume: float) -> float:
    """
    Calculate liquidity score based on volume.

    Higher volume relative to average = higher score.

    Args:
        volume: Current volume
        avg_volume: Average volume (e.g., 30-day average)

    Returns:
        Score between 0.0 and 1.0
    """
    if avg_volume == 0:
        return 0.0
    
    ratio = volume / avg_volume
    # 1.0x = 0.5, 2.0x = 0.75, 3.0x+ = 1.0
    return normalize_score(ratio, 0.0, 3.0)


def volatility_score(volatility: float, min_vol: float = 0.1, max_vol: float = 0.5) -> float:
    """
    Calculate volatility score.

    Moderate volatility is preferred (too low = boring, too high = risky).

    Args:
        volatility: Current volatility (e.g., 30-day std dev)
        min_vol: Minimum acceptable volatility
        max_vol: Maximum acceptable volatility

    Returns:
        Score between 0.0 and 1.0 (peaks at moderate volatility)
    """
    if volatility < min_vol:
        return normalize_score(volatility, 0.0, min_vol)
    elif volatility > max_vol:
        # Invert: higher volatility = lower score
        return 1.0 - normalize_score(volatility, max_vol, max_vol * 2)
    else:
        # Sweet spot: moderate volatility
        return normalize_score(volatility, min_vol, max_vol)


def momentum_score(price_change_pct: float, min_change: float = 0.02, max_change: float = 0.10) -> float:
    """
    Calculate momentum score based on price change.

    Positive momentum is preferred, but extreme moves may be overbought.

    Args:
        price_change_pct: Price change percentage (e.g., 5-day return)
        min_change: Minimum change to consider (e.g., 2%)
        max_change: Maximum change before considering overbought (e.g., 10%)

    Returns:
        Score between 0.0 and 1.0
    """
    if price_change_pct < min_change:
        return normalize_score(price_change_pct, -max_change, min_change)
    elif price_change_pct > max_change:
        # Invert: extreme moves may be overbought
        return 1.0 - normalize_score(price_change_pct, max_change, max_change * 2)
    else:
        # Sweet spot: positive momentum
        return normalize_score(price_change_pct, min_change, max_change)

