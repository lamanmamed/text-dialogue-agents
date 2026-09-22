import tempfile
import unittest
from pathlib import Path

from text_dialogue_agents.restaurant_agent import RestaurantBackend, normalize_slot, AREA_MAP


class RestaurantTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.backend = RestaurantBackend(
            [
                {"id": "r001", "name": "Centre Noodle House", "area": "centre", "food": "chinese", "pricerange": "cheap"},
                {"id": "r002", "name": "North Curry", "area": "north", "food": "indian", "pricerange": "moderate"},
            ],
            {"r001": [{"name": "Spring Rolls", "price": 5.0}]},
            Path(self.temp.name) / "orders.json",
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_slot_normalization(self):
        self.assertEqual(normalize_slot("central area", AREA_MAP), "centre")

    def test_search(self):
        result = self.backend.search_restaurants(area="central", food="chinese", pricerange="budget")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["results"][0]["id"], "r001")

    def test_order_lifecycle(self):
        draft = self.backend.create_order("r001", [{"name": "spring rolls", "quantity": 3}], "Laman")
        self.assertTrue(draft["ok"])
        order_id = draft["order"]["order_id"]
        self.assertEqual(draft["order"]["total_price"], 15.0)
        confirmed = self.backend.finalize_order(order_id)
        self.assertEqual(confirmed["order"]["status"], "confirmed")
        cancelled = self.backend.cancel_order(order_id)
        self.assertEqual(cancelled["order"]["status"], "cancelled")
        self.assertEqual(self.backend.get_order(order_id)["order"]["status"], "cancelled")

    def test_bad_order_id_is_rejected(self):
        result = self.backend.get_order("123")
        self.assertFalse(result["ok"])
        self.assertIn("ORD-123456", result["error"])


if __name__ == "__main__":
    unittest.main()
