"""
Unit tests for weather tool language detection.
"""

import pytest
from bot.llm.tools.weather import _detect_language


class TestDetectLanguage:
    """Test language detection from city names."""
    
    def test_chinese_traditional(self):
        """Test Traditional Chinese cities."""
        assert _detect_language("台北") == "zh"
        assert _detect_language("板橋") == "zh"
        assert _detect_language("桃園") == "zh"
        assert _detect_language("高雄") == "zh"
    
    def test_chinese_simplified(self):
        """Test Simplified Chinese cities."""
        assert _detect_language("北京") == "zh"
        assert _detect_language("上海") == "zh"
        assert _detect_language("广州") == "zh"
        assert _detect_language("深圳") == "zh"
    
    def test_japanese_hiragana_katakana(self):
        """Test Japanese cities using Hiragana/Katakana (distinct from Chinese)."""
        # Hiragana
        assert _detect_language("とうきょう") == "ja"
        # Katakana
        assert _detect_language("トウキョウ") == "ja"
        # Mix of Katakana + Kanji
        assert _detect_language("東京") == "zh"  # Kanji overlaps with Chinese - detected as zh (acceptable)
    
    def test_japanese_romaji(self):
        """Test Japanese cities using Roman letters (common in practice)."""
        # These are Latin letters, so detected as English (correct!)
        assert _detect_language("Tokyo") == "en"
        assert _detect_language("Osaka") == "en"
        assert _detect_language("Kyoto") == "en"
    
    def test_korean(self):
        """Test Korean cities."""
        assert _detect_language("서울") == "ko"
        assert _detect_language("부산") == "ko"
        assert _detect_language("인천") == "ko"
    
    def test_russian(self):
        """Test Russian cities."""
        assert _detect_language("Москва") == "ru"
        assert _detect_language("Санкт-Петербург") == "ru"
    
    def test_arabic(self):
        """Test Arabic city names."""
        assert _detect_language("القاهرة") == "ar"
        assert _detect_language("الرياض") == "ar"
    
    def test_thai(self):
        """Test Thai cities."""
        assert _detect_language("กรุงเทพ") == "th"
        assert _detect_language("เชียงใหม่") == "th"
    
    def test_hebrew(self):
        """Test Hebrew cities."""
        assert _detect_language("תל אביב") == "he"
        assert _detect_language("ירושלים") == "he"
    
    def test_english_default(self):
        """Test English city names (default)."""
        assert _detect_language("New York") == "en"
        assert _detect_language("London") == "en"
        assert _detect_language("Paris") == "en"
        assert _detect_language("Sydney") == "en"
    
    def test_mixed_english(self):
        """Test English cities with spaces."""
        assert _detect_language("Los Angeles") == "en"
        assert _detect_language("San Francisco") == "en"
        assert _detect_language("Hong Kong") == "en"
    
    def test_numbers_ignored(self):
        """Test that numbers don't affect language detection (uses Latin chars)."""
        # Tokyo123 has no CJK characters → detected as English
        assert _detect_language("Tokyo123") == "en"
        # City100 has no CJK characters → detected as English  
        assert _detect_language("City100") == "en"
        # Numbers with Chinese chars should still detect Chinese
        assert _detect_language("台北123") == "zh"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])