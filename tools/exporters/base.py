"""
tools/exporters/base.py

BaseExporter ABC — 所有模型导出器的公共接口。

权重去重策略：每张图导出后**立即**剥离权重到 weights.bin，再继续下一张。
峰值磁盘 ≈ weights.bin（一份权重）+ 当前图的 fat ONNX（一份权重），
而不是所有图同时在磁盘上。
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple

from aria.models.base import FrameworkConfig

logger = logging.getLogger(__name__)

WEIGHTS_FILE = "weights.bin"


class BaseExporter(ABC):
    def __init__(self, cfg: FrameworkConfig, model_path: str):
        self.cfg        = cfg
        self.model_path = model_path
        self._model     = None

    # ------------------------------------------------------------------
    # 子类必须实现
    # ------------------------------------------------------------------

    @abstractmethod
    def load_model(self) -> None:
        """加载 HF 模型到 self._model，dtype=float16。"""

    @abstractmethod
    def export_prefill(self, out_dir: str, seq_len: int) -> str:
        """导出 prefill_{seq_len}.onnx，返回路径。"""

    @abstractmethod
    def export_decode(self, out_dir: str) -> str:
        """
        导出单张 decode.onnx，返回路径。

        「固定 max buffer + 偏移」设计：decode 不再按 kv_len 分 bucket，
        kv_cache 输入恒为 max_seq_len 长度，自回归每步只变 position_id /
        attention_mask（数据），图本身唯一。
        """

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    def export_all(self, out_dir: str) -> List[str]:
        """
        导出全部图。每张图导出后立即剥离权重，不等所有图都生成。

        峰值磁盘占用：
          weights.bin（一份权重）+ 当前图的 fat ONNX（一份权重）
        """
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        if self._model is None:
            logger.info("[export] 加载模型: %s", self.model_path)
            self.load_model()

        # offsets: {init_name: (offset_in_weights_bin, byte_length)}
        # 在所有图之间共享，保证同名权重只写一次
        offsets: Dict[str, Tuple[int, int]] = {}
        paths: List[str] = []

        for seq_len in self.cfg.llm.prefill_buckets:
            logger.info("[export] prefill_%d", seq_len)
            p = self.export_prefill(out_dir, seq_len)
            _strip_weights(p, out_dir, offsets)
            paths.append(p)

        logger.info("[export] decode (single graph, max_seq_len=%d)",
                    self.cfg.llm.max_seq_len)
        p = self.export_decode(out_dir)
        _strip_weights(p, out_dir, offsets)
        paths.append(p)

        weights_path = Path(out_dir) / WEIGHTS_FILE
        logger.info(
            "[export] 完成：%d 张图，weights.bin %.1f GiB，共 %d 个 tensor",
            len(paths),
            weights_path.stat().st_size / 1024 ** 3,
            len(offsets),
        )
        return paths

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _onnx_path(self, out_dir: str, name: str) -> str:
        return str(Path(out_dir) / f"{name}.onnx")


# ---------------------------------------------------------------------------
# 即时权重剥离
# ---------------------------------------------------------------------------

def _strip_weights(
    onnx_path: str,
    out_dir:   str,
    offsets:   Dict[str, Tuple[int, int]],
) -> None:
    """
    从 onnx_path 中剥离所有 initializer 的数据：
    - 尚未见过的 initializer → 追加到 weights.bin，记录 offset
    - 已见过的 → 直接复用已有 offset（各 bucket 同名权重值相同）
    然后用只含 graph 拓扑的精简版覆盖原文件。

    每次调用只需要一张 fat ONNX 在磁盘上，处理完立即替换成 lean 版本。
    """
    try:
        import onnx
        from onnx import numpy_helper, TensorProto, external_data_helper
    except ImportError:
        logger.warning("[strip] onnx 未安装，跳过权重剥离（图含完整权重）")
        return

    weights_path = Path(out_dir) / WEIGHTS_FILE

    model = onnx.load(onnx_path, load_external_data=True)

    # 追加写 weights.bin（已有内容不重写）
    with open(weights_path, "ab") as wf:
        for init in model.graph.initializer:
            if init.name in offsets:
                continue
            raw    = numpy_helper.to_array(init).tobytes()
            offset = wf.seek(0, 2)          # 当前文件末尾
            wf.write(raw)
            offsets[init.name] = (offset, len(raw))

    # 把 initializer 改成外部引用，清空内联数据
    for init in model.graph.initializer:
        if init.name not in offsets:
            continue
        off, length = offsets[init.name]
        external_data_helper.set_external_data(
            init,
            location = WEIGHTS_FILE,
            offset   = off,
            length   = length,
        )
        init.data_location = TensorProto.EXTERNAL
        for field in ("raw_data", "float_data", "int32_data", "int64_data",
                      "double_data", "uint64_data", "string_data"):
            init.ClearField(field)

    onnx.save(model, onnx_path)

    graph_kb = Path(onnx_path).stat().st_size / 1024
    weights_gib = weights_path.stat().st_size / 1024 ** 3
    logger.info(
        "[strip] %-28s graph=%.1f KiB  weights.bin=%.2f GiB",
        Path(onnx_path).name, graph_kb, weights_gib,
    )
