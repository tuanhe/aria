"""
core/scheduler.py

三级流水线调度器：
  Thread-1 视觉预处理
  Thread-2 LLM推理（Prefill + Decode）
  Thread-3 输出消费（动作执行 / 文本回调）

设计目标：掩盖各阶段延迟，提高端到端吞吐。
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

_SENTINEL = object()  # 终止信号


@dataclass
class VisionTask:
    request_id: int
    image:      np.ndarray
    text:       Any           # 原始文本（字符串或message列表）
    extra:      dict          = None  # 附加信息（session_id等）


@dataclass
class PrefillTask:
    request_id:  int
    vision_feat: np.ndarray
    text:        Any
    extra:       dict = None


@dataclass
class OutputTask:
    request_id: int
    result:     Any           # 动作数组 或 文本字符串


class PipelineScheduler:
    """
    三级异步流水线。

    用法：
        scheduler = PipelineScheduler(
            vision_fn   = lambda task: ...,   # 返回 vision_feat
            inference_fn = lambda task: ...,  # 返回 result
            output_fn   = lambda task: ...,   # 消费result
        )
        scheduler.start()
        scheduler.submit(image, text)
        scheduler.stop()
    """

    def __init__(self,
                 vision_fn:    Callable[[VisionTask], np.ndarray],
                 inference_fn: Callable[[PrefillTask], Any],
                 output_fn:    Callable[[OutputTask], None],
                 queue_size:   int = 4):

        self._vision_fn    = vision_fn
        self._inference_fn = inference_fn
        self._output_fn    = output_fn

        self._vision_q    = queue.Queue(maxsize=queue_size)
        self._prefill_q   = queue.Queue(maxsize=queue_size)
        self._output_q    = queue.Queue(maxsize=queue_size)

        self._threads: list = []
        self._running = False
        self._req_id  = 0
        self._lock    = threading.Lock()

    def start(self) -> None:
        self._running = True
        self._threads = [
            threading.Thread(target=self._vision_worker,    name="VisionWorker",    daemon=True),
            threading.Thread(target=self._inference_worker, name="InferenceWorker", daemon=True),
            threading.Thread(target=self._output_worker,    name="OutputWorker",    daemon=True),
        ]
        for t in self._threads:
            t.start()
        logger.info("[ARIA/Scheduler] 流水线已启动（3线程）")

    def stop(self, timeout: float = 5.0) -> None:
        # 向每个队列发送终止信号
        self._vision_q.put(_SENTINEL)
        for t in self._threads:
            t.join(timeout=timeout)
        self._running = False
        logger.info("[ARIA/Scheduler] 流水线已停止")

    def submit(self,
               image: np.ndarray,
               text:  Any,
               extra: dict = None) -> int:
        """提交一个推理请求，返回request_id"""
        with self._lock:
            req_id = self._req_id
            self._req_id += 1

        task = VisionTask(request_id=req_id, image=image, text=text, extra=extra or {})
        self._vision_q.put(task)
        logger.debug(f"[ARIA/Scheduler] 提交请求 req_id={req_id}")
        return req_id

    # ------------------------------------------------------------------
    # 工作线程
    # ------------------------------------------------------------------

    def _vision_worker(self) -> None:
        logger.debug("[ARIA/VisionWorker] 启动")
        while True:
            task = self._vision_q.get()
            if task is _SENTINEL:
                self._prefill_q.put(_SENTINEL)
                break
            try:
                t0          = time.perf_counter()
                vision_feat = self._vision_fn(task)
                elapsed     = (time.perf_counter() - t0) * 1000
                logger.debug(f"[ARIA/VisionWorker] req={task.request_id} 耗时={elapsed:.1f}ms")

                self._prefill_q.put(PrefillTask(
                    request_id  = task.request_id,
                    vision_feat = vision_feat,
                    text        = task.text,
                    extra       = task.extra,
                ))
            except Exception as e:
                logger.error(f"[ARIA/VisionWorker] req={task.request_id} 异常: {e}", exc_info=True)

    def _inference_worker(self) -> None:
        logger.debug("[ARIA/InferenceWorker] 启动")
        while True:
            task = self._prefill_q.get()
            if task is _SENTINEL:
                self._output_q.put(_SENTINEL)
                break
            try:
                t0      = time.perf_counter()
                result  = self._inference_fn(task)
                elapsed = (time.perf_counter() - t0) * 1000
                logger.debug(f"[ARIA/InferenceWorker] req={task.request_id} 耗时={elapsed:.1f}ms")

                self._output_q.put(OutputTask(
                    request_id = task.request_id,
                    result     = result,
                ))
            except Exception as e:
                logger.error(f"[ARIA/InferenceWorker] req={task.request_id} 异常: {e}", exc_info=True)

    def _output_worker(self) -> None:
        logger.debug("[ARIA/OutputWorker] 启动")
        while True:
            task = self._output_q.get()
            if task is _SENTINEL:
                break
            try:
                self._output_fn(task)
            except Exception as e:
                logger.error(f"[ARIA/OutputWorker] req={task.request_id} 异常: {e}", exc_info=True)
