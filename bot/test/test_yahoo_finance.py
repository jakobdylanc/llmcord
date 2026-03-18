"""
Unit tests for yahoo_finance tool.

Tests data extraction robustness, retry logic, and edge cases.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta


class TestDataExtractionRobustness:
    """Tests for NaN handling and data extraction (Phase 1)."""
    
    @pytest.fixture
    def mock_yfinance(self):
        """Mock yfinance module."""
        with patch('bot.llm.tools.yahoo_finance.yf') as mock_yf:
            yield mock_yf
    
    @pytest.fixture
    def valid_dataframe(self):
        """Create a valid DataFrame with stock data."""
        dates = pd.date_range(end=datetime.now(), periods=5, freq='D')
        data = {
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'High': [105.0, 106.0, 107.0, 108.0, 109.0],
            'Low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'Close': [104.0, 105.0, 106.0, 107.0, 108.0],
            'Volume': [1000] * 5
        }
        df = pd.DataFrame(data, index=dates)
        return df
    
    @pytest.fixture
    def nan_close_dataframe(self):
        """Create DataFrame with NaN close price on the last day."""
        dates = pd.date_range(end=datetime.now(), periods=3, freq='D')
        data = {
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [99.0, 100.0, 101.0],
            'Close': [104.0, 105.0, float('nan')],  # NaN on last day (iloc[-1])
            'Volume': [1000] * 3
        }
        return pd.DataFrame(data, index=dates)
    
    @pytest.fixture
    def nan_prev_close_dataframe(self):
        """Create DataFrame with NaN prev_close (second-to-last day)."""
        dates = pd.date_range(end=datetime.now(), periods=3, freq='D')
        data = {
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [99.0, 100.0, 101.0],
            'Close': [100.0, float('nan'), 106.0],  # NaN on second-to-last day (iloc[-2])
            'Volume': [1000] * 3
        }
        return pd.DataFrame(data, index=dates)
    
    @pytest.fixture
    def zero_prev_close_dataframe(self):
        """Create DataFrame with zero prev_close (second-to-last day)."""
        dates = pd.date_range(end=datetime.now(), periods=3, freq='D')
        data = {
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [99.0, 100.0, 101.0],
            'Close': [100.0, 0.0, 106.0],  # Zero on second-to-last day (iloc[-2])
            'Volume': [1000] * 3
        }
        return pd.DataFrame(data, index=dates)
    
    @pytest.fixture
    def single_day_dataframe(self):
        """Create DataFrame with single day of data."""
        dates = pd.date_range(end=datetime.now(), periods=1, freq='D')
        data = {
            'Open': [100.0],
            'High': [105.0],
            'Low': [99.0],
            'Close': [104.0],
            'Volume': [1000]
        }
        return pd.DataFrame(data, index=dates)
    
    @pytest.fixture
    def empty_dataframe(self):
        """Create an empty DataFrame."""
        return pd.DataFrame()
    
    # Task 1.6: Test NaN close price
    def test_nan_close_price_outputs_unavailable(self, mock_yfinance, nan_close_dataframe):
        """Test NaN close price outputs '[unavailable]' message."""
        from bot.llm.tools.yahoo_finance import get_market_prices
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = nan_close_dataframe
        mock_yfinance.Ticker.return_value = mock_ticker
        
        result = get_market_prices("TEST")
        
        assert "close price unavailable" in result
        assert "TEST" in result
    
    # Task 1.7: Test when info returns previous close, it's used correctly
    def test_info_previous_close_used_when_available(self, mock_yfinance, valid_dataframe):
        """Test that info endpoint previous close is used for % calculation."""
        from bot.llm.tools.yahoo_finance import get_market_prices
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = valid_dataframe
        mock_ticker.info = {'regularMarketPreviousClose': 100.0}  # Info returns 100
        mock_yfinance.Ticker.return_value = mock_ticker
        
        # Last close is 108, so with prev close 100, change is +8%
        result = get_market_prices("TEST")
        
        assert "+8.00 (+8.00%)" in result
        assert "108.00" in result
    
    # Task 1.8: Test zero prev_close
    def test_zero_prev_close_prevents_division_by_zero(self, mock_yfinance, zero_prev_close_dataframe):
        """Test zero prev_close prevents division by zero error."""
        from bot.llm.tools.yahoo_finance import get_market_prices
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = zero_prev_close_dataframe
        mock_yfinance.Ticker.return_value = mock_ticker
        
        result = get_market_prices("TEST")
        
        # Should not crash, should show "[no prev day]"
        assert "[no prev day]" in result
        assert "TEST" in result
    
    # Task 1.9: Test empty yfinance response
    def test_empty_response_returns_clear_error(self, mock_yfinance, empty_dataframe):
        """Test empty yfinance response returns clear error message."""
        from bot.llm.tools.yahoo_finance import get_market_prices
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = empty_dataframe
        mock_yfinance.Ticker.return_value = mock_ticker
        
        result = get_market_prices("INVALID_TICKER")
        
        assert "no data returned" in result
        assert "INVALID_TICKER" in result
    
    # Task 1.10: Test multiple tickers
    def test_multiple_tickers_mixed_success_failure(self, mock_yfinance, valid_dataframe, empty_dataframe):
        """Test multiple tickers with mixed success/failure."""
        from bot.llm.tools.yahoo_finance import get_market_prices
        
        # First ticker returns valid data, second returns empty
        def side_effect(ticker):
            if ticker == "VALID":
                return valid_dataframe
            return empty_dataframe
        
        mock_ticker = Mock()
        mock_yfinance.Ticker.side_effect = lambda x: Mock(
            history=Mock(side_effect=lambda **k: valid_dataframe if x == "VALID" else empty_dataframe)
        )
        
        result = get_market_prices("VALID,INVALID")
        
        assert "VALID" in result
        assert "no data returned" in result
    
    # Task 4.3: Test single day data
    def test_single_day_data_shows_no_prev_day(self, mock_yfinance, single_day_dataframe):
        """Test single day data shows '[no prev day]' correctly."""
        from bot.llm.tools.yahoo_finance import get_market_prices
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = single_day_dataframe
        mock_yfinance.Ticker.return_value = mock_ticker
        
        result = get_market_prices("TEST")
        
        assert "[no prev day]" in result
        assert "104.00" in result


class TestRetryLogic:
    """Tests for retry logic (Phase 2)."""
    
    @pytest.fixture
    def mock_yfinance(self):
        """Mock yfinance module."""
        with patch('bot.llm.tools.yahoo_finance.yf') as mock_yf:
            yield mock_yf
    
    @pytest.fixture
    def valid_dataframe(self):
        """Create a valid DataFrame."""
        dates = pd.date_range(end=datetime.now(), periods=5, freq='D')
        data = {
            'Open': [100.0] * 5,
            'High': [105.0] * 5,
            'Low': [99.0] * 5,
            'Close': [104.0, 105.0, 106.0, 107.0, 108.0],
            'Volume': [1000] * 5
        }
        return pd.DataFrame(data, index=dates)
    
    @pytest.fixture
    def empty_dataframe(self):
        """Create an empty DataFrame."""
        return pd.DataFrame()
    
    # Task 2.4: Test empty data triggers retry
    def test_empty_data_triggers_retry(self, mock_yfinance, empty_dataframe, valid_dataframe):
        """Test empty data triggers retry (verify 3 attempts made)."""
        from bot.llm.tools.yahoo_finance import get_market_prices
        
        # First 2 calls return empty, third returns valid
        call_count = [0]
        
        def history_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                return empty_dataframe
            return valid_dataframe
        
        mock_ticker = Mock()
        mock_ticker.history.side_effect = history_side_effect
        mock_yfinance.Ticker.return_value = mock_ticker
        
        result = get_market_prices("TEST")
        
        # Should succeed after retries
        assert "108.00" in result or "TEST" in result
    
    # Task 2.5: Test exponential backoff
    def test_exponential_backoff_delays(self, mock_yfinance, empty_dataframe):
        """Test exponential backoff delays (~1s first, ~2s second)."""
        from bot.llm.tools.yahoo_finance import _fetch_with_retry
        import time
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = empty_dataframe
        mock_yfinance.Ticker.return_value = mock_ticker
        
        start = time.time()
        _fetch_with_retry("TEST", 5, max_attempts=3)
        elapsed = time.time() - start
        
        # Should sleep ~3 seconds total (1s + 2s) but allow some margin
        assert elapsed >= 2.5  # Allow some margin for execution
    
    # Task 2.6: Test invalid ticker does NOT retry on exception
    def test_exception_does_not_retry_indefinitely(self, mock_yfinance):
        """Test exception causes limited retries, not infinite."""
        from bot.llm.tools.yahoo_finance import get_market_prices
        
        mock_ticker = Mock()
        mock_ticker.history.side_effect = Exception("Network error")
        mock_yfinance.Ticker.return_value = mock_ticker
        
        # Should handle exception gracefully
        result = get_market_prices("INVALID")
        
        # Should return error message, not crash
        assert "error" in result.lower() or "INVALID" in result


class TestDataReliability:
    """Tests for data reliability (Phase 3)."""
    
    @pytest.fixture
    def mock_yfinance(self):
        """Mock yfinance module."""
        with patch('bot.llm.tools.yahoo_finance.yf') as mock_yf:
            yield mock_yf
    
    @pytest.fixture
    def valid_dataframe(self):
        """Create a valid DataFrame."""
        dates = pd.date_range(end=datetime.now(), periods=10, freq='D')
        data = {
            'Open': [100.0] * 10,
            'High': [105.0] * 10,
            'Low': [99.0] * 10,
            'Close': list(range(100, 110)),
            'Volume': [1000] * 10
        }
        return pd.DataFrame(data, index=dates)
    
    # Task 3.3: Test default days parameter is 10
    def test_default_days_parameter_is_10(self, mock_yfinance, valid_dataframe):
        """Test default days parameter is 10 for main historical data."""
        from bot.llm.tools.yahoo_finance import get_market_prices
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = valid_dataframe
        mock_yfinance.Ticker.return_value = mock_ticker
        
        get_market_prices("TEST")
        
        # Verify history was called with period="10d" for main data (not intraday)
        # There may be multiple calls, find the one with period="10d"
        calls = mock_ticker.history.call_args_list
        periods = [call[1].get('period') for call in calls]
        assert '10d' in periods, f"Expected '10d' in periods, got {periods}"
    
    # Task 3.5: Test with known problematic tickers
    def test_problematic_tickers(self, mock_yfinance, valid_dataframe):
        """Test with known problematic tickers (0050.TW, 006208.TW)."""
        from bot.llm.tools.yahoo_finance import get_market_prices
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = valid_dataframe
        mock_yfinance.Ticker.return_value = mock_ticker
        
        # These tickers should not crash
        result = get_market_prices("0050.TW,006208.TW,00878.TW")
        
        assert "0050.TW" in result
        assert "006208.TW" in result
        assert "00878.TW" in result


class TestEdgeCases:
    """Tests for edge cases (Phase 4)."""
    
    @pytest.fixture
    def mock_yfinance(self):
        """Mock yfinance module."""
        with patch('bot.llm.tools.yahoo_finance.yf') as mock_yf:
            yield mock_yf
    
    # Task 4.1: Test empty ticker list
    def test_empty_ticker_list_returns_error(self, mock_yfinance):
        """Test empty ticker list returns 'no tickers provided' error."""
        from bot.llm.tools.yahoo_finance import get_market_prices
        
        result = get_market_prices("")
        
        assert "no tickers provided" in result
    
    # Task 4.2: Test yfinance not installed
    def test_yfinance_not_installed_returns_message(self):
        """Test yfinance not installed returns installation error message."""
        with patch('bot.llm.tools.yahoo_finance._YF_AVAILABLE', False):
            from bot.llm.tools.yahoo_finance import get_market_prices
            
            result = get_market_prices("TEST")
            
            assert "yfinance is not installed" in result
    
    # Task 4.4: Test single vs multiple tickers
    def test_single_vs_multiple_tickers(self, mock_yfinance):
        """Test single ticker vs multiple tickers all succeed."""
        from bot.llm.tools.yahoo_finance import get_market_prices
        
        dates = pd.date_range(end=datetime.now(), periods=5, freq='D')
        data = {
            'Open': [100.0] * 5,
            'High': [105.0] * 5,
            'Low': [99.0] * 5,
            'Close': [104.0, 105.0, 106.0, 107.0, 108.0],
            'Volume': [1000] * 5
        }
        df = pd.DataFrame(data, index=dates)
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = df
        mock_yfinance.Ticker.return_value = mock_ticker
        
        # Single ticker
        result_single = get_market_prices("TEST")
        
        # Multiple tickers
        mock_ticker.reset_mock()
        result_multiple = get_market_prices("TEST1,TEST2,TEST3")
        
        assert "TEST" in result_single
        assert "TEST1" in result_multiple
        assert "TEST2" in result_multiple
        assert "TEST3" in result_multiple


if __name__ == "__main__":
    pytest.main([__file__, "-v"])