"""Run the restaurant backend without a language model."""

from text_dialogue_agents.restaurant_agent import RestaurantBackend

restaurants = [
    {"id": "r001", "name": "Centre Noodle House", "area": "centre", "food": "chinese", "pricerange": "cheap"},
    {"id": "r002", "name": "Riverside Pasta", "area": "riverside", "food": "italian", "pricerange": "moderate"},
]
menus = {
    "r001": [
        {"name": "Spring Rolls", "price": 5.0},
        {"name": "Fried Rice", "price": 7.5},
    ]
}
backend = RestaurantBackend(restaurants, menus)
print(backend.search_restaurants(area="central", food="Chinese", pricerange="budget"))
booking = backend.book_restaurant("r001", "8pm", 3, "Laman", day="Friday")
print(booking)
draft = backend.create_order("r001", [{"name": "Spring Rolls", "quantity": 3}], "Laman", booking=booking["booking"])
print(draft)
print(backend.finalize_order(draft["order"]["order_id"]))
