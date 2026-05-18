"""ARIA CLI 入口。

通过 pyproject 暴露为 `aria` 命令；同时支持 `python -m aria`。
"""

import argparse
import logging
import os
import sys

import numpy as np

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("aria.cli")


def _build_executor(name: str, cfg=None):
    from aria.backends import build_executor
    if name == "mock":
        return build_executor("mock", latency_ms=5.0)
    if name == "torch":
        if cfg is None:
            raise ValueError("torch backend 需要 FrameworkConfig")
        return build_executor("torch", config=cfg)
    return build_executor(name)


def run_vla(config_path: str, executor_name: str = "mock", graph_dir: str = None):
    from aria.models.base     import FrameworkConfig
    from aria.runtime.vla_runtime import VLARuntime

    cfg      = FrameworkConfig.from_yaml(config_path)
    if graph_dir:
        cfg.graph_dir = graph_dir
    executor = _build_executor(executor_name, cfg=cfg)
    runtime  = VLARuntime.from_config(cfg, executor)

    rng = np.random.default_rng(0)
    h, w = cfg.vision.resolution
    logger.info("开始VLA推理循环（按Ctrl+C退出）")

    for step in range(5):
        image       = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        instruction = "pick up the red cube and place it in the bin"

        action = runtime.infer(image, instruction)

        if cfg.action.head_type == "flow_matching":
            logger.info(f"Step {step}: action[0]={action[0].round(3)}")
        else:
            logger.info(f"Step {step}: action={action.round(3)}")

    executor.enable_profiling(False)
    stats = executor.get_profiling_stats()
    if stats:
        print("\n[性能统计]")
        for name, s in stats.items():
            print(f"  {name}: mean={s['mean']:.1f}ms p95={s['p95']:.1f}ms")


def run_vlm(config_path: str, executor_name: str = "mock", graph_dir: str = None):
    from aria.models.base     import FrameworkConfig
    from aria.runtime.vlm_runtime import VLMRuntime

    cfg      = FrameworkConfig.from_yaml(config_path)
    if graph_dir:
        cfg.graph_dir = graph_dir
    executor = _build_executor(executor_name, cfg=cfg)
    runtime  = VLMRuntime.from_config(cfg, executor)

    rng   = np.random.default_rng(0)
    h, w  = cfg.vision.resolution
    image = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)

    logger.info("--- 单轮对话示例 ---")
    response = runtime.chat([
        {"role": "user", "content": [
            {"type": "image", "data": image},
            {"type": "text",  "data": "describe this image in detail"},
        ]}
    ])
    logger.info(f"Response: '{response}'")

    logger.info("\n--- 多轮对话示例 ---")
    session_id = runtime.new_session()

    turns = [
        [{"role": "user", "content": [
            {"type": "image", "data": image},
            {"type": "text",  "data": "what objects are in this image"},
        ]}],
        [{"role": "user", "content": "which one is the largest"}],
        [{"role": "user", "content": "describe its color"}],
    ]

    for i, msgs in enumerate(turns):
        r = runtime.chat(msgs, session_id=session_id)
        logger.info(f"Turn {i+1} response: '{r}'")

    runtime.close_session(session_id)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="aria", description="ARIA 推理框架示例")
    parser.add_argument("--config", required=True,
                        help="配置文件路径（yaml）")
    from aria.backends import list_executors
    parser.add_argument("--executor", default="mock",
                        choices=list_executors(),
                        help="执行器后端（mock = 纯 numpy；其他按已注册的后端）")
    parser.add_argument("--graph-dir", default=None,
                        help="覆盖配置里的 graph_dir，用来切换不同的 engine 集合")
    args = parser.parse_args(argv)

    cfg_path = os.path.abspath(args.config)
    if not os.path.exists(cfg_path):
        logger.error(f"配置文件不存在: {cfg_path}")
        sys.exit(1)

    import yaml
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    model_mode = raw.get("mode", "vla")

    logger.info(f"配置: {cfg_path}  model_mode: {model_mode}")

    if model_mode == "vla":
        run_vla(cfg_path, args.executor, args.graph_dir)
    elif model_mode == "vlm":
        run_vlm(cfg_path, args.executor, args.graph_dir)
    else:
        logger.error(f"未知mode: {model_mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
