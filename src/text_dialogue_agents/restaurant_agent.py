"""Structured restaurant, booking, and order actions for a tool-using dialogue agent."""

from __future__ import annotations

import json
import random
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

AREA_MAP = {
    "centre": "centre", "center": "centre", "city centre": "centre", "city center": "centre",
    "downtown": "centre", "central": "centre", "north": "north", "south": "south",
    "east": "east", "west": "west", "riverside": "riverside", "river side": "riverside",
}
FOOD_MAP = {
    "chinese": "chinese", "italian": "italian", "indian": "indian", "japanese": "japanese",
    "vegetarian": "vegetarian", "veggie": "vegetarian", "british": "british", "english": "british",
}
PRICE_MAP = {
    "cheap": "cheap", "inexpensive": "cheap", "budget": "cheap", "low": "cheap",
    "moderate": "moderate", "mid": "moderate", "medium": "moderate",
    "expensive": "expensive", "pricey": "expensive", "high": "expensive", "high-end": "expensive",
}


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    value = re.sub(r"\s+", " ", str(value).strip().lower())
    return value or None


def normalize_slot(value: Any, mapping: dict[str, str]) -> str | None:
    text = _normalize_text(value)
    if not text:
        return None
    if text in mapping:
        return mapping[text]
    for phrase, canonical in mapping.items():
        if phrase in text:
            return canonical
    return None


class RestaurantBackend:
    """Deterministic APIs that the language model can call instead of inventing booking data."""

    def __init__(self, restaurants: list[dict], menus: dict[str, list[dict]], order_path: str | Path | None = None):
        self.restaurants = restaurants
        self.by_id = {str(item["id"]): item for item in restaurants}
        self.menus = menus
        self.order_path = Path(order_path) if order_path is not None else None
        self.orders = self._load_orders()
        self.pending: dict[str, dict] = {}

    def _load_orders(self) -> dict[str, dict]:
        if self.order_path is None or not self.order_path.exists():
            return {}
        try:
            data = json.loads(self.order_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_orders(self) -> None:
        if self.order_path is not None:
            self.order_path.write_text(json.dumps(self.orders, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")

    def _new_id(self, prefix: str) -> str:
        existing = set(self.orders) | set(self.pending)
        while True:
            value = f"{prefix}-{random.randint(100000, 999999)}"
            if value not in existing:
                return value

    def search_restaurants(
        self,
        *,
        area: str | None = None,
        food: str | None = None,
        pricerange: str | None = None,
        name: str | None = None,
        limit: int = 5,
    ) -> dict:
        area = normalize_slot(area, AREA_MAP) if area else None
        food = normalize_slot(food, FOOD_MAP) if food else None
        pricerange = normalize_slot(pricerange, PRICE_MAP) if pricerange else None
        name = _normalize_text(name)
        results = []
        for restaurant in self.restaurants:
            if area and _normalize_text(restaurant.get("area")) != area:
                continue
            if food and _normalize_text(restaurant.get("food") or restaurant.get("cuisine")) != food:
                continue
            if pricerange and _normalize_text(restaurant.get("pricerange")) != pricerange:
                continue
            if name and name not in (_normalize_text(restaurant.get("name")) or ""):
                continue
            results.append(restaurant)
        limit = max(1, min(int(limit), 10))
        return {"ok": True, "count": len(results), "results": results[:limit]}

    def book_restaurant(self, restaurant_id: str, time: str, people: int, customer_name: str, day: str | None = None) -> dict:
        restaurant_id = str(restaurant_id).strip()
        if restaurant_id not in self.by_id:
            return {"ok": False, "error": f"Unknown restaurant_id: {restaurant_id}"}
        try:
            people = int(people)
        except (TypeError, ValueError):
            return {"ok": False, "error": "people must be an integer"}
        if not 1 <= people <= 20:
            return {"ok": False, "error": "people must be between 1 and 20"}
        if not str(time).strip() or not str(customer_name).strip():
            return {"ok": False, "error": "time and customer_name are required"}
        booking = {
            "booking_id": self._new_id("BKG"),
            "restaurant_id": restaurant_id,
            "restaurant_name": self.by_id[restaurant_id].get("name"),
            "day": str(day).strip() if day else "unspecified_day",
            "time": str(time).strip(),
            "people": people,
            "customer_name": str(customer_name).strip(),
            "created_at": self._now(),
            "status": "confirmed",
        }
        return {"ok": True, "booking": booking}

    def list_menu(self, restaurant_id: str, limit: int = 20) -> dict:
        restaurant_id = str(restaurant_id).strip()
        if restaurant_id not in self.by_id:
            return {"ok": False, "error": f"Unknown restaurant_id: {restaurant_id}"}
        menu = self.menus.get(restaurant_id, [])
        items = [
            {"name": item.get("name"), "price": float(item.get("price", 0.0))}
            for item in menu[: max(1, min(int(limit), 50))]
        ]
        return {
            "ok": True,
            "restaurant_id": restaurant_id,
            "restaurant_name": self.by_id[restaurant_id].get("name"),
            "count": len(items),
            "items": items,
        }

    def _find_menu_item(self, restaurant_id: str, item_name: str) -> dict | None:
        target = _normalize_text(item_name)
        for item in self.menus.get(restaurant_id, []):
            name = _normalize_text(item.get("name"))
            if name == target or (name and target and (target in name or name in target)):
                return item
        return None

    def create_order(self, restaurant_id: str, items: list[dict], customer_name: str, booking: dict | None = None) -> dict:
        restaurant_id = str(restaurant_id).strip()
        if restaurant_id not in self.by_id:
            return {"ok": False, "error": f"Unknown restaurant_id: {restaurant_id}"}
        if not items:
            return {"ok": False, "error": "items must be a non-empty list"}
        resolved = []
        total = 0.0
        for raw in items:
            name = raw.get("name") or raw.get("item")
            if not name:
                return {"ok": False, "error": "Each item must include a name"}
            try:
                quantity = int(raw.get("quantity", 1))
            except (TypeError, ValueError):
                return {"ok": False, "error": f"Invalid quantity for item '{name}'"}
            if not 1 <= quantity <= 50:
                return {"ok": False, "error": f"Quantity for '{name}' must be between 1 and 50"}
            menu_item = self._find_menu_item(restaurant_id, str(name))
            if menu_item is None:
                return {"ok": False, "error": f"Menu item not found: {name}"}
            unit = float(menu_item.get("price", 0.0))
            line = round(unit * quantity, 2)
            total += line
            resolved.append({"name": menu_item.get("name"), "quantity": quantity, "unit_price": unit, "line_total": line})
        order_id = self._new_id("ORD")
        now = self._now()
        order = {
            "order_id": order_id,
            "restaurant_id": restaurant_id,
            "restaurant_name": self.by_id[restaurant_id].get("name"),
            "customer_name": str(customer_name).strip(),
            "items": resolved,
            "total_price": round(total, 2),
            "status": "draft",
            "created_at": now,
            "updated_at": now,
            "booking": booking,
        }
        self.pending[order_id] = order
        return {"ok": True, "order": order}

    def finalize_order(self, order_id: str) -> dict:
        order_id = str(order_id).strip()
        if order_id not in self.pending:
            return {"ok": False, "error": f"Unknown pending order_id: {order_id}"}
        order = self.pending.pop(order_id)
        order["status"] = "confirmed"
        order["updated_at"] = self._now()
        self.orders[order_id] = order
        self._save_orders()
        return {"ok": True, "order": order}

    def validate_order_id(self, order_id: str) -> dict:
        order_id = str(order_id).strip()
        valid = bool(re.fullmatch(r"ORD-\d{6}", order_id))
        return {
            "ok": valid,
            "valid_format": valid,
            "exists": order_id in self.orders or order_id in self.pending,
            "order_id": order_id,
            "error": None if valid else "Order number must match the format ORD-123456",
        }

    def get_order(self, order_id: str) -> dict:
        order_id = str(order_id).strip()
        check = self.validate_order_id(order_id)
        if not check["valid_format"]:
            return {"ok": False, "error": check["error"], "order_id": order_id}
        if order_id in self.orders:
            return {"ok": True, "order": self.orders[order_id]}
        if order_id in self.pending:
            return {"ok": True, "order": self.pending[order_id]}
        return {"ok": False, "error": f"Order number does not exist: {order_id}", "order_id": order_id}

    def cancel_order(self, order_id: str) -> dict:
        order_id = str(order_id).strip()
        target = self.orders if order_id in self.orders else self.pending if order_id in self.pending else None
        if target is None:
            return {"ok": False, "error": f"Order number does not exist: {order_id}"}
        order = target[order_id]
        if order.get("status") == "cancelled":
            return {"ok": True, "message": f"Order {order_id} is already cancelled", "order": order}
        order["status"] = "cancelled"
        order["updated_at"] = self._now()
        if target is self.orders:
            self._save_orders()
        return {"ok": True, "order": order}

    def execute(self, tool_name: str, arguments: dict | None = None) -> dict:
        arguments = arguments or {}
        tools = {
            "search_restaurants": self.search_restaurants,
            "book_restaurant": self.book_restaurant,
            "list_menu": self.list_menu,
            "create_order": self.create_order,
            "finalize_order": self.finalize_order,
            "get_order": self.get_order,
            "cancel_order": self.cancel_order,
            "validate_order_id": self.validate_order_id,
        }
        if tool_name not in tools:
            return {"ok": False, "error": f"Unknown tool: {tool_name}"}
        try:
            return tools[tool_name](**arguments)
        except TypeError as exc:
            return {"ok": False, "error": str(exc)}


SYSTEM_PROMPT = """You are a task-oriented restaurant assistant.
You can help users find restaurants, make bookings, view menus, place orders, check orders, and cancel orders.
When an action requires backend data, respond with one JSON tool call instead of inventing a result.
Ask for missing required information before calling a tool.
After a tool result is provided, explain it to the user naturally and concisely.
"""


def extract_json(text: str) -> dict | None:
    """Extract one JSON object from a model response."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class OllamaRestaurantAgent:
    """Small JSON-action loop around a local Ollama chat model."""

    def __init__(
        self,
        backend: RestaurantBackend,
        *,
        model: str = "llama3.1:8b-instruct-q4_K_M",
        url: str = "http://localhost:11434/api/chat",
        timeout: int = 180,
    ):
        self.backend = backend
        self.model = model
        self.url = url
        self.timeout = timeout
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _chat(self) -> str:
        response = requests.post(
            self.url,
            json={"model": self.model, "messages": self.messages, "stream": False, "options": {"temperature": 0.0}},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def turn(self, user_text: str, *, max_hops: int = 8) -> str:
        self.messages.append({"role": "user", "content": user_text})
        for _ in range(max_hops):
            reply = self._chat()
            action = extract_json(reply)
            if not action or "tool" not in action:
                self.messages.append({"role": "assistant", "content": reply})
                return reply
            result = self.backend.execute(action["tool"], action.get("arguments", {}))
            self.messages.append({"role": "assistant", "content": reply})
            self.messages.append({"role": "user", "content": "Tool result: " + json.dumps(result)})
        raise RuntimeError("Agent exceeded max_hops without producing a user-facing response")
