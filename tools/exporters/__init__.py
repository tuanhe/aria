from __future__ import annotations

import importlib
from typing import Dict, Tuple

from tools.exporters.base import BaseExporter

_EXPORTERS: Dict[str, Tuple[str, str]] = {
    "qwen3": ("tools.exporters.qwen3", "Qwen3Exporter"),
}


def list_exporters():
    return list(_EXPORTERS.keys())


def build_exporter(name: str, **kwargs) -> BaseExporter:
    if name not in _EXPORTERS:
        raise ValueError(f"未知 exporter '{name}'，可选: {list_exporters()}")
    mod_path, cls_name = _EXPORTERS[name]
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)(**kwargs)
