"""
EnsembleCombiner - Merges results from multiple symbol pickers.

Combines MetricPicker, FuzzyPicker, and LLMPicker results using weighted voting.
Deduplicates symbols and produces a final ranked list.
"""

from collections import defaultdict
from typing import List

from src.config import settings
from src.services.discovery.pickers.base import PickerResult
from src.utils.logging import get_logger

log = get_logger(__name__)


class EnsembleCombiner:
    """
    Combines results from multiple symbol pickers using weighted voting.

    For each symbol:
    1. Collects scores from all pickers that found it
    2. Computes weighted average: sum(score * weight) / sum(weights)
    3. Ranks symbols by final composite score
    4. Returns deduplicated, ranked list
    """

    def __init__(
        self,
        metric_weight: float | None = None,
        fuzzy_weight: float | None = None,
        llm_weight: float | None = None,
    ):
        """
        Initialize EnsembleCombiner with picker weights.

        Args:
            metric_weight: Weight for MetricPicker (default: from config)
            fuzzy_weight: Weight for FuzzyPicker (default: from config)
            llm_weight: Weight for LLMPicker (default: from config)
        """
        self.metric_weight = metric_weight or settings.discovery.metric_weight
        self.fuzzy_weight = fuzzy_weight or settings.discovery.fuzzy_weight
        self.llm_weight = llm_weight or settings.discovery.llm_weight

        # Normalize weights to sum to 1.0
        total_weight = self.metric_weight + self.fuzzy_weight + self.llm_weight
        if total_weight > 0:
            self.metric_weight /= total_weight
            self.fuzzy_weight /= total_weight
            self.llm_weight /= total_weight
        else:
            # Default equal weights if all zero
            self.metric_weight = 1.0 / 3.0
            self.fuzzy_weight = 1.0 / 3.0
            self.llm_weight = 1.0 / 3.0

        # Map picker names to weights
        self.picker_weights = {
            "metric": self.metric_weight,
            "fuzzy": self.fuzzy_weight,
            "llm": self.llm_weight,
        }

    def combine(
        self,
        metric_results: List[PickerResult] | None = None,
        fuzzy_results: List[PickerResult] | None = None,
        llm_results: List[PickerResult] | None = None,
    ) -> List[PickerResult]:
        """
        Combine results from multiple pickers using weighted voting.

        Args:
            metric_results: Results from MetricPicker (optional)
            fuzzy_results: Results from FuzzyPicker (optional)
            llm_results: Results from LLMPicker (optional)

        Returns:
            List of PickerResult objects, sorted by composite score (highest first)
        """
        # Collect all results by picker
        all_results = []
        if metric_results:
            all_results.extend(("metric", r) for r in metric_results)
        if fuzzy_results:
            all_results.extend(("fuzzy", r) for r in fuzzy_results)
        if llm_results:
            all_results.extend(("llm", r) for r in llm_results)

        if not all_results:
            log.warning("ensemble_no_results")
            return []

        log.info(
            "ensemble_combining",
            metric_count=len(metric_results) if metric_results else 0,
            fuzzy_count=len(fuzzy_results) if fuzzy_results else 0,
            llm_count=len(llm_results) if llm_results else 0,
        )

        # Group by symbol (deduplication)
        symbol_scores: dict[str, list[tuple[str, float]]] = defaultdict(list)
        symbol_reasons: dict[str, list[str]] = defaultdict(list)
        symbol_metadata: dict[str, dict] = defaultdict(dict)

        for picker_name, result in all_results:
            symbol = result.symbol.upper()
            symbol_scores[symbol].append((picker_name, result.score))
            symbol_reasons[symbol].append(f"{picker_name}: {result.reason}")
            # Merge metadata
            if result.metadata:
                symbol_metadata[symbol].update(result.metadata)
                symbol_metadata[symbol].setdefault("pickers", []).append(picker_name)

        # Compute weighted composite scores
        final_results = []
        for symbol, picker_scores in symbol_scores.items():
            # Calculate weighted average
            weighted_sum = 0.0
            total_weight = 0.0

            for picker_name, score in picker_scores:
                weight = self.picker_weights.get(picker_name, 0.0)
                weighted_sum += score * weight
                total_weight += weight

            if total_weight > 0:
                composite_score = weighted_sum / total_weight
            else:
                composite_score = 0.0

            # Combine reasons
            combined_reason = " | ".join(symbol_reasons[symbol])

            # Ensure metadata includes picker list
            metadata = symbol_metadata[symbol]
            if "pickers" not in metadata:
                metadata["pickers"] = [picker_name for picker_name, _ in picker_scores]

            final_results.append(
                PickerResult(
                    symbol=symbol,
                    score=composite_score,
                    reason=combined_reason,
                    metadata=metadata,
                )
            )

        # Sort by composite score (highest first)
        final_results.sort(key=lambda x: x.score, reverse=True)

        log.info(
            "ensemble_complete",
            final_count=len(final_results),
            unique_symbols=len(symbol_scores),
        )

        return final_results

    def combine_from_dict(
        self,
        picker_results: dict[str, List[PickerResult]],
    ) -> List[PickerResult]:
        """
        Combine results from a dictionary of picker results.

        Convenience method for when results are already grouped by picker name.

        Args:
            picker_results: Dict mapping picker names to their results
                Example: {"metric": [...], "fuzzy": [...], "llm": [...]}

        Returns:
            List of PickerResult objects, sorted by composite score
        """
        return self.combine(
            metric_results=picker_results.get("metric"),
            fuzzy_results=picker_results.get("fuzzy"),
            llm_results=picker_results.get("llm"),
        )

