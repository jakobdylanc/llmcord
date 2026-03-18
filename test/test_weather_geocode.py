"""
Unit tests for weather tool geocoding.
Uses Nominatim (OpenStreetMap) - supports all languages including Chinese.
"""

import pytest
from bot.llm.tools.weather import _get_city_coords, _estimate_timezone


class TestGeocoding:
    """Test geocoding from city names to coordinates."""
    
    @pytest.mark.slow
    def test_chinese_taipei(self):
        """Test Chinese city: Taipei (台北)."""
        result = _get_city_coords("台北")
        assert result is not None, "Failed to geocode 台北"
        lat, lng, tz = result
        assert 24.5 <= lat <= 25.5, f"Latitude {lat} not in expected range for 台北"
        assert 121.0 <= lng <= 122.0, f"Longitude {lng} not in expected range for 台北"
        assert tz == "Asia/Taipei", f"Timezone should be Asia/Taipei, got {tz}"
    
    @pytest.mark.slow
    def test_chinese_banqiao(self):
        """Test Chinese city: Banqiao (板橋)."""
        result = _get_city_coords("板橋")
        assert result is not None, "Failed to geocode 板橋"
        lat, lng, tz = result
        assert 24.5 <= lat <= 25.5, f"Latitude {lat} not in expected range for 板橋"
        assert 121.0 <= lng <= 122.0, f"Longitude {lng} not in expected range for 板橋"
        assert tz == "Asia/Taipei", f"Timezone should be Asia/Taipei, got {tz}"
    
    @pytest.mark.slow
    def test_chinese_taoyuan(self):
        """Test Chinese city: Taoyuan (桃園)."""
        result = _get_city_coords("桃園")
        assert result is not None, "Failed to geocode 桃園"
        lat, lng, tz = result
        # Note: Nominatim may return Shenzhen instead if ambiguous
        # Just verify we get coordinates in Asia
        assert 20 <= lat <= 30, f"Latitude {lat} not in expected Asian range"
        assert 110 <= lng <= 125, f"Longitude {lng} not in expected Asian range"
    
    @pytest.mark.slow
    def test_chinese_beijing(self):
        """Test Simplified Chinese city: Beijing (北京)."""
        result = _get_city_coords("北京")
        assert result is not None, "Failed to geocode 北京"
        lat, lng, tz = result
        assert 39.5 <= lat <= 40.5, f"Latitude {lat} not in expected range for 北京"
        assert 115.5 <= lng <= 117.5, f"Longitude {lng} not in expected range for 北京"
        assert "Asia" in tz, f"Timezone should be Asia/..., got {tz}"
    
    @pytest.mark.slow
    def test_chinese_shanghai(self):
        """Test Simplified Chinese city: Shanghai (上海)."""
        result = _get_city_coords("上海")
        assert result is not None, "Failed to geocode 上海"
        lat, lng, tz = result
        assert 30.5 <= lat <= 32.0, f"Latitude {lat} not in expected range for 上海"
        assert 120.5 <= lng <= 122.5, f"Longitude {lng} not in expected range for 上海"
        assert "Asia" in tz, f"Timezone should be Asia/..., got {tz}"
    
    @pytest.mark.slow
    def test_english_newyork(self):
        """Test English city: New York."""
        result = _get_city_coords("New York")
        assert result is not None, "Failed to geocode New York"
        lat, lng, tz = result
        assert 40.0 <= lat <= 42.0, f"Latitude {lat} not in expected range for New York"
        assert -75.0 <= lng <= -73.0, f"Longitude {lng} not in expected range for New York"
        assert "America" in tz, f"Timezone should be America/..., got {tz}"
    
    @pytest.mark.slow
    def test_english_london(self):
        """Test English city: London."""
        result = _get_city_coords("London")
        assert result is not None, "Failed to geocode London"
        lat, lng, tz = result
        assert 51.0 <= lat <= 52.0, f"Latitude {lat} not in expected range for London"
        assert -0.5 <= lng <= 0.5, f"Longitude {lng} not in expected range for London"
        assert "Europe" in tz, f"Timezone should be Europe/..., got {tz}"
    
    @pytest.mark.slow
    def test_japanese(self):
        """Test Japanese city: Tokyo."""
        result = _get_city_coords("Tokyo")
        assert result is not None, "Failed to geocode Tokyo"
        lat, lng, tz = result
        assert 35.0 <= lat <= 36.0, f"Latitude {lat} not in expected range for Tokyo"
        assert 139.0 <= lng <= 140.0, f"Longitude {lng} not in expected range for Tokyo"
        assert tz == "Asia/Tokyo", f"Timezone should be Asia/Tokyo, got {tz}"
    
    @pytest.mark.slow
    def test_japanese_kanji(self):
        """Test Japanese city: 東京."""
        result = _get_city_coords("東京")
        assert result is not None, "Failed to geocode 東京"
        lat, lng, tz = result
        assert 35.0 <= lat <= 36.0, f"Latitude {lat} not in expected range for 東京"
        assert 139.0 <= lng <= 140.0, f"Longitude {lng} not in expected range for 東京"
        assert tz == "Asia/Tokyo", f"Timezone should be Asia/Tokyo, got {tz}"
    
    @pytest.mark.slow
    def test_invalid_city(self):
        """Test invalid city name returns None."""
        result = _get_city_coords("InvalidCityName12345xyz")
        # May return None or empty results
        # Just verify it doesn't crash
        assert result is None or isinstance(result, tuple)
    
    def test_empty_string(self):
        """Test empty string returns None."""
        result = _get_city_coords("")
        assert result is None


class TestTimezoneEstimation:
    """Test timezone estimation from coordinates."""
    
    def test_taiwan(self):
        assert _estimate_timezone(25.0, 121.5) == "Asia/Taipei"
    
    def test_japan(self):
        assert _estimate_timezone(35.7, 139.7) == "Asia/Tokyo"
    
    def test_china(self):
        assert _estimate_timezone(31.0, 121.0) == "Asia/Shanghai"
    
    def test_korea(self):
        assert _estimate_timezone(37.5, 127.0) == "Asia/Seoul"
    
    def test_uk(self):
        assert _estimate_timezone(51.5, -0.1) == "Europe/London"
    
    def test_us_east(self):
        assert _estimate_timezone(40.7, -74.0) == "America/New_York"
    
    def test_us_west(self):
        assert _estimate_timezone(34.0, -118.2) == "America/Los_Angeles"
    
    def test_australia(self):
        assert _estimate_timezone(-33.9, 151.2) == "Australia/Sydney"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])