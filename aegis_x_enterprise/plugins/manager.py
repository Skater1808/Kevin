"""Dynamic plugin loader.

Scans the ``plugins/`` directory at boot, dynamically imports every ``*.py``
module via :mod:`importlib`, instantiates each :class:`~tools.base.BasePlugin`
subclass it finds, and lets the plugin register its tools.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import logging
from pathlib import Path

from tools.base import BasePlugin
from tools.registry import ToolRegistry

logger = logging.getLogger("aegis.plugins")

PLUGIN_DIR: Path = Path(__file__).resolve().parent


class PluginManager:
    """Discovers and loads plugins from the plugin directory."""

    def __init__(self, registry: ToolRegistry, plugin_dir: Path | None = None) -> None:
        self.registry = registry
        self.plugin_dir = plugin_dir or PLUGIN_DIR
        self.loaded: list[BasePlugin] = []

    async def discover(self) -> list[BasePlugin]:
        """Asynchronously scan and load all plugins. Returns the loaded plugins."""
        return await asyncio.to_thread(self._discover_sync)

    def _discover_sync(self) -> list[BasePlugin]:
        self.loaded.clear()
        if not self.plugin_dir.is_dir():
            logger.warning("Plugin directory does not exist: %s", self.plugin_dir)
            return self.loaded

        for file in sorted(self.plugin_dir.glob("*.py")):
            if file.name in {"__init__.py", "manager.py"}:
                continue
            self._load_file(file)

        logger.info("Loaded %d plugin(s): %s", len(self.loaded), [p.name for p in self.loaded])
        return self.loaded

    def _load_file(self, file: Path) -> None:
        module_name = f"aegis_plugins.{file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, file)
        if spec is None or spec.loader is None:
            logger.error("Could not create import spec for plugin: %s", file)
            return
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:  # noqa: BLE001 - a broken plugin must not crash boot
            logger.exception("Failed to import plugin module: %s", file)
            return

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BasePlugin) and obj is not BasePlugin and obj.__module__ == module.__name__:
                try:
                    plugin = obj()
                    plugin.register_tools(self.registry)
                    self.loaded.append(plugin)
                    logger.info("Activated plugin '%s' from %s", plugin.name, file.name)
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to initialize plugin class %s", obj.__name__)
