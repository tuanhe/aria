"""
models/vision_encoder.py

视觉编码器模块。
端侧固定分辨率场景：只编译一张静态图，直接调用。
"""

from __future__ import annotations

import logging

import numpy as np

from aria.core.executor import NPUExecutor, GraphMeta
from aria.models.base import FrameworkConfig

logger = logging.getLogger(__name__)

GRAPH_NAME = "vision_encoder"


class VisionEncoder:
    """
    视觉编码器推理封装。

    端侧固定分辨率：分辨率在配置中指定，tile数固定，只需一张编译图。
    输出：视觉特征 [batch, total_vision_tokens, feat_dim]
    """

    def __init__(self, config: FrameworkConfig, executor: NPUExecutor):
        self.config   = config
        self.executor = executor
        self.vcfg     = config.vision

        # 注册NPU图
        # 真实部署时，path指向编译好的.bin/.om/.rknn文件
        # Mock模式下文件不存在没关系，MockExecutor不真正读文件
        meta = GraphMeta(
            name  = GRAPH_NAME,
            path  = f"{config.graph_dir}/vision_encoder.bin",
            input_shapes = {
                # [batch, num_tiles, C, H, W]
                "tiles": (
                    config.max_batch,
                    self.vcfg.num_tiles,
                    self.vcfg.channels,
                    self.vcfg.tile_size[0],
                    self.vcfg.tile_size[1],
                ),
            },
            output_shapes = {
                # [batch, total_vision_tokens, feat_dim]
                "vision_feat": (
                    config.max_batch,
                    self.vcfg.total_vision_tokens,
                    self.vcfg.feat_dim,
                ),
            },
            output_dtypes = {"vision_feat": np.float16},
        )
        executor.register_graph(meta)

        logger.info(
            f"[ARIA/Vision] 分辨率={self.vcfg.resolution} "
            f"tiles={self.vcfg.num_tiles} "
            f"vision_tokens={self.vcfg.total_vision_tokens}"
        )

    def encode(self, image: np.ndarray) -> np.ndarray:
        """
        编码单张图像。
        image: [H, W, C] uint8 或 [C, H, W] float32
        返回: [1, total_vision_tokens, feat_dim] float16
        """
        tiles = self._preprocess(image)
        out   = self.executor.run(GRAPH_NAME, {"tiles": tiles})
        return out["vision_feat"]

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理：resize → tile切分 → 归一化 → 组成batch
        返回 [1, num_tiles, C, tile_h, tile_w]
        """
        # 如果是HWC格式，转CHW
        if image.ndim == 3 and image.shape[2] in (1, 3):
            image = image.transpose(2, 0, 1)  # HWC → CHW

        # Mock模式下直接生成正确shape的随机数据
        C, H, W = image.shape
        th, tw  = self.vcfg.tile_size
        nh      = self.vcfg.resolution[0] // th
        nw      = self.vcfg.resolution[1] // tw

        # 简单resize（实际部署建议用cv2或libjpeg-turbo）
        # 这里用numpy插值模拟
        tiles = np.zeros((self.vcfg.num_tiles, C, th, tw), dtype=np.float32)

        tile_idx = 0
        for i in range(nh):
            for j in range(nw):
                # 切tile（真实场景应先resize到目标分辨率再切）
                src_h_start = int(i * H / nh)
                src_h_end   = int((i + 1) * H / nh)
                src_w_start = int(j * W / nw)
                src_w_end   = int((j + 1) * W / nw)
                tile = image[:, src_h_start:src_h_end, src_w_start:src_w_end]

                # 简单resize（实际用interpolation）
                # 这里直接pad/crop到tile_size
                tile_resized = np.zeros((C, th, tw), dtype=np.float32)
                h_copy = min(tile.shape[1], th)
                w_copy = min(tile.shape[2], tw)
                tile_resized[:, :h_copy, :w_copy] = tile[:, :h_copy, :w_copy]
                tiles[tile_idx] = tile_resized
                tile_idx += 1

        # 归一化到 [-1, 1]
        tiles = (tiles / 127.5 - 1.0).astype(np.float16)

        # 加batch维度
        return tiles[np.newaxis, :]  # [1, num_tiles, C, th, tw]
