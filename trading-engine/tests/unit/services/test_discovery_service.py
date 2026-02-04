"""
Unit tests for SymbolDiscoveryService.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.discovery.pickers.base import PickerResult
from src.services.discovery.service import SymbolDiscoveryService


class TestSymbolDiscoveryService:
    """Test SymbolDiscoveryService orchestration."""

    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        session = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        return session

    @pytest.fixture
    def service(self, mock_session):
        """Create SymbolDiscoveryService with mocked session."""
        with patch("src.services.discovery.service.MetricPicker"), patch(
            "src.services.discovery.service.FuzzyPicker"
        ), patch("src.services.discovery.service.LLMPicker"):
            return SymbolDiscoveryService(db_session=mock_session)

    @pytest.mark.asyncio
    async def test_run_discovery_cycle_no_pickers(self, mock_session):
        """Test discovery cycle with no enabled pickers."""
        with patch("src.config.settings.discovery.enabled_pickers", []):
            service = SymbolDiscoveryService(db_session=mock_session)
            result = await service.run_discovery_cycle()

            assert result["discovered_symbols"] == []
            assert result["picker_suggestions"] == []
            assert result["ensemble_results"] == []
            assert result["watchlist_updates"] == 0

    @pytest.mark.asyncio
    async def test_run_discovery_cycle_with_results(self, mock_session):
        """Test discovery cycle with picker results."""
        # Mock pickers
        metric_picker = AsyncMock()
        metric_picker.name = "metric"
        metric_picker.pick = AsyncMock(
            return_value=[
                PickerResult(symbol="AAPL", score=1.0, reason="Passed filters"),
                PickerResult(symbol="MSFT", score=1.0, reason="Passed filters"),
            ]
        )

        fuzzy_picker = AsyncMock()
        fuzzy_picker.name = "fuzzy"
        fuzzy_picker.pick = AsyncMock(
            return_value=[
                PickerResult(symbol="AAPL", score=0.8, reason="High liquidity"),
            ]
        )

        llm_picker = AsyncMock()
        llm_picker.name = "llm"
        llm_picker.pick = AsyncMock(
            return_value=[
                PickerResult(symbol="GOOGL", score=0.7, reason="LLM recommendation"),
            ]
        )

        with patch("src.services.discovery.service.MetricPicker", return_value=metric_picker), patch(
            "src.services.discovery.service.FuzzyPicker", return_value=fuzzy_picker
        ), patch("src.services.discovery.service.LLMPicker", return_value=llm_picker), patch(
            "src.config.settings.discovery.enabled_pickers", ["metric", "fuzzy", "llm"]
        ), patch(
            "src.config.settings.discovery.max_watchlist_size", 20
        ):
            service = SymbolDiscoveryService(db_session=mock_session)

            # Mock database operations
            # Mock the result chain: session.execute() -> result.scalars().all()
            mock_scalars_result = MagicMock()
            mock_scalars_result.all.return_value = []  # Empty watchlist initially
            
            # Result for getting current watchlist (has scalars().all())
            mock_watchlist_result = MagicMock()
            mock_watchlist_result.scalars.return_value = mock_scalars_result
            
            # Result for checking existing entries (has scalar_one_or_none())
            mock_existing_result = MagicMock()
            mock_existing_result.scalar_one_or_none.return_value = None  # No existing entries
            
            # Count execute() calls:
            # 1. Get current watchlist (1 call)
            # 2. _persist_discovered_symbols: check duplicates for each (symbol, picker) pair
            #    - MSFT from metric, AAPL from metric, AAPL from fuzzy, GOOGL from llm = 4 calls
            # 3. _persist_picker_suggestions: check duplicates for each (symbol, picker) pair
            #    - Same 4 calls
            # 4. _update_watchlist: check existing entries for each symbol
            #    - MSFT, AAPL, GOOGL = 3 calls
            # Total: 1 + 4 + 4 + 3 = 12 calls
            mock_session.execute = AsyncMock(side_effect=[
                mock_watchlist_result,  # 1. Get current watchlist
                # 2. _persist_discovered_symbols deduplication (4 calls)
                mock_existing_result,   # MSFT from metric
                mock_existing_result,   # AAPL from metric
                mock_existing_result,   # AAPL from fuzzy
                mock_existing_result,   # GOOGL from llm
                # 3. _persist_picker_suggestions deduplication (4 calls)
                mock_existing_result,   # MSFT from metric
                mock_existing_result,   # AAPL from metric
                mock_existing_result,   # AAPL from fuzzy
                mock_existing_result,   # GOOGL from llm
                # 4. _update_watchlist existing checks (3 calls)
                mock_existing_result,   # Check MSFT
                mock_existing_result,   # Check AAPL
                mock_existing_result,   # Check GOOGL
            ])
            mock_session.add = MagicMock()

            result = await service.run_discovery_cycle()

            assert len(result["discovered_symbols"]) > 0
            assert len(result["picker_suggestions"]) > 0
            assert len(result["ensemble_results"]) > 0
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_discovery_cycle_picker_error(self, mock_session):
        """Test discovery cycle handles picker errors gracefully."""
        metric_picker = AsyncMock()
        metric_picker.name = "metric"
        metric_picker.pick = AsyncMock(side_effect=Exception("Picker error"))

        with patch("src.services.discovery.service.MetricPicker", return_value=metric_picker), patch(
            "src.config.settings.discovery.enabled_pickers", ["metric"]
        ):
            service = SymbolDiscoveryService(db_session=mock_session)

            result = await service.run_discovery_cycle()

            # Should continue despite error
            assert "discovered_symbols" in result
            assert "ensemble_results" in result

    @pytest.mark.asyncio
    async def test_get_active_watchlist(self, mock_session):
        """Test getting active watchlist."""
        from src.database.models import Watchlist
        from datetime import datetime

        # Mock watchlist entries
        mock_watchlist = [
            MagicMock(symbol="AAPL", added_at=datetime.now()),
            MagicMock(symbol="MSFT", added_at=datetime.now()),
        ]

        mock_scalars_result = MagicMock()
        mock_scalars_result.all.return_value = mock_watchlist
        
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars_result
        mock_session.execute = AsyncMock(return_value=mock_result)

        service = SymbolDiscoveryService(db_session=mock_session)
        watchlist = await service.get_active_watchlist(mock_session)

        assert watchlist == ["AAPL", "MSFT"]

    @pytest.mark.asyncio
    async def test_remove_from_watchlist(self, mock_session):
        """Test removing symbol from watchlist."""
        from src.database.models import Watchlist
        from datetime import datetime

        # Mock existing watchlist entry
        mock_entry = MagicMock()
        mock_entry.active = True
        mock_entry.removed_at = None

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_entry
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        service = SymbolDiscoveryService(db_session=mock_session)
        removed = await service.remove_from_watchlist("AAPL", mock_session)

        assert removed is True
        assert mock_entry.active is False
        assert mock_entry.removed_at is not None
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_from_watchlist_not_found(self, mock_session):
        """Test removing non-existent symbol from watchlist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        service = SymbolDiscoveryService(db_session=mock_session)
        removed = await service.remove_from_watchlist("INVALID", mock_session)

        assert removed is False

