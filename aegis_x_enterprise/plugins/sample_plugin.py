"""Sample plugin: a small weather lookup tool.

Demonstrates the plugin contract. It registers a ``weather`` tool that queries
the free, key-less Open-Meteo API and returns the current temperature for a
given latitude/longitude.
"""

from __future__ import annotations

from typing import Any

import httpx

from tools.base import BasePlugin, BaseTool, ToolResult
from tools.registry import ToolRegistry


class WeatherTool(BaseTool):
    name = "weather"
    description = "Return the current temperature for a latitude/longitude pair."
    parameters = {
        "latitude": "Latitude in decimal degrees",
        "longitude": "Longitude in decimal degrees",
    }

    async def run(self, **kwargs: Any) -> ToolResult:
        try:
            latitude = float(kwargs["latitude"])
            longitude = float(kwargs["longitude"])
        except (KeyError, TypeError, ValueError) as exc:
            return ToolResult(ok=False, output="", error=f"Invalid coordinates: {exc}")

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m",
        }
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:  # noqa: BLE001
            return ToolResult(ok=False, output="", error=f"Weather lookup failed: {exc}")

        current = data.get("current", {})
        temperature = current.get("temperature_2m")
        unit = data.get("current_units", {}).get("temperature_2m", "°C")
        return ToolResult(
            ok=True,
            output=f"Current temperature at ({latitude}, {longitude}): {temperature}{unit}",
            data={"temperature": temperature, "unit": unit},
        )


class WeatherPlugin(BasePlugin):
    name = "weather_plugin"
    description = "Provides a current-weather lookup tool via Open-Meteo."

    def register_tools(self, registry: ToolRegistry) -> None:
        registry.register(WeatherTool())
