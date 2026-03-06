import unittest
from bot.llm.tools.web_search import brave_web_search


class TestBraveWebSearch(unittest.TestCase):
    def test_brave_web_search(self):
        query = "how's the weather for new taipei"
        result = brave_web_search(query)

        # Top-level keys from Brave API
        self.assertIn("type", result, "Missing 'type' key")
        self.assertEqual(result["type"], "search")

        self.assertIn("query", result, "Missing 'query' key")
        self.assertIn("original", result["query"])
        self.assertEqual(result["query"]["original"], query)

        # Web results
        self.assertIn("web", result, "Missing 'web' key")
        self.assertIn("results", result["web"], "Missing 'web.results' key")
        self.assertIsInstance(result["web"]["results"], list)
        self.assertGreater(len(result["web"]["results"]), 0, "Expected at least one result")

        # Spot-check first result shape
        first = result["web"]["results"][0]
        self.assertIn("title", first)
        self.assertIn("url", first)
        self.assertIn("description", first)


if __name__ == "__main__":
    unittest.main()