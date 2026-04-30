"""Helpers for optional heavyweight imports."""

from __future__ import annotations

from importlib import import_module


def require(module_name: str, install_hint: str | None = None):
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        hint = install_hint or "Install the project dependencies first."
        raise RuntimeError(f"Missing required dependency '{module_name}'. {hint}") from exc
