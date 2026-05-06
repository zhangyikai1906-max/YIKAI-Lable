# -*- coding: utf-8 -*-
"""
GroundingDINO-SwinT 封装。
适配 RTX 2060 6GB：使用 SwinT（Tiny）骨干，推理 VRAM ~1.2 GB。

依赖安装：
    pip install git+https://github.com/IDEA-Research/GroundingDINO.git

权重下载（见 scripts/download_weights.py）：
    - GroundingDINO_SwinT_OGC.py   (config)
    - groundingdino_swint_ogc.pth  (694 MB)
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


class DetectionResult:
    __slots__ = ("label", "score", "box_xyxy", "box_xywh")

    def __init__(self, label: str, score: float, box_xyxy: List[float]):
        self.label = label
        self.score = score
        # [x1, y1, x2, y2] 像素坐标（绝对值）
        self.box_xyxy = box_xyxy
        x1, y1, x2, y2 = box_xyxy
        self.box_xywh = [x1, y1, x2 - x1, y2 - y1]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "score": self.score,
            "box_xyxy": self.box_xyxy,
            "box_xywh": self.box_xywh,
        }


def _build_prompt(classes: List[str]) -> str:
    """
    GroundingDINO 使用句点分隔多类别，末尾需要句点。
    例：["car", "person"] → "car . person ."
    """
    cleaned = [c.strip().lower() for c in classes if c.strip()]
    return " . ".join(cleaned) + " ." if cleaned else ""


class GroundingDINOEngine:
    """
    GroundingDINO-SwinT 推理引擎（线程安全单例）。

    使用方式：
        engine = GroundingDINOEngine()
        engine.load(config_path, checkpoint_path)
        results = engine.detect(image, ["landslide", "crack"])
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._transform = None
        self.load_error: Optional[str] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def is_ready(self) -> bool:
        return self._model is not None

    def load(self, config_path: str, checkpoint_path: str) -> Dict:
        """加载模型权重，返回 {"ok": bool, "message": str}。"""
        with self._lock:
            self.load_error = None
            try:
                from groundingdino.util.inference import load_model
                from groundingdino.datasets.transforms import Compose
            except ImportError:
                self.load_error = (
                    "未安装 GroundingDINO，请执行：\n"
                    "pip install git+https://github.com/IDEA-Research/GroundingDINO.git"
                )
                return {"ok": False, "message": self.load_error}

            config_path = str(config_path)
            checkpoint_path = str(checkpoint_path)

            if not Path(config_path).is_file():
                self.load_error = f"GroundingDINO 配置文件不存在: {config_path}"
                return {"ok": False, "message": self.load_error}

            if not Path(checkpoint_path).is_file():
                self.load_error = f"GroundingDINO 权重文件不存在: {checkpoint_path}"
                return {"ok": False, "message": self.load_error}

            try:
                model = load_model(config_path, checkpoint_path, device=self._device)
                model.eval()
                self._model = model
            except Exception as e:
                self._model = None
                self.load_error = f"GroundingDINO 加载失败: {e}"
                return {"ok": False, "message": self.load_error}

            return {"ok": True, "message": f"GroundingDINO 加载成功（{self._device}）"}

    def detect(
        self,
        image: Image.Image,
        classes: List[str],
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> List[DetectionResult]:
        """
        对单张图片执行开放词汇检测。

        Args:
            image:          PIL RGB 图像
            classes:        待检测类别名列表，例如 ["car", "person"]
            box_threshold:  检测框置信度阈值（建议 0.25~0.40）
            text_threshold: 文本匹配阈值（建议 0.20~0.30）

        Returns:
            DetectionResult 列表
        """
        if not self.is_ready():
            raise RuntimeError("GroundingDINO 模型未加载，请先调用 load()")

        text_prompt = _build_prompt(classes)
        if not text_prompt:
            return []

        width, height = image.size

        with self._lock:
            try:
                from groundingdino.util.inference import predict
                import groundingdino.datasets.transforms as T

                transform = T.Compose([
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
                img_tensor, _ = transform(image, None)

                with torch.inference_mode():
                    boxes, logits, phrases = predict(
                        model=self._model,
                        image=img_tensor,
                        caption=text_prompt,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold,
                        device=self._device,
                    )

            except Exception as e:
                raise RuntimeError(f"GroundingDINO 推理失败: {e}") from e

        results: List[DetectionResult] = []
        if boxes is None or len(boxes) == 0:
            return results

        # boxes 格式为 cx,cy,w,h（归一化），转换为像素 xyxy
        boxes_np = boxes.cpu().numpy()
        logits_np = logits.cpu().numpy()

        for i, (box, score, phrase) in enumerate(zip(boxes_np, logits_np, phrases)):
            cx, cy, w, h = box
            x1 = float((cx - w / 2) * width)
            y1 = float((cy - h / 2) * height)
            x2 = float((cx + w / 2) * width)
            y2 = float((cy + h / 2) * height)

            x1 = max(0.0, x1)
            y1 = max(0.0, y1)
            x2 = min(float(width), x2)
            y2 = min(float(height), y2)

            # phrase 可能是 "car" 或 "car person"，尝试匹配最近的类别
            matched_label = _match_label(phrase, classes)

            results.append(DetectionResult(
                label=matched_label,
                score=float(score),
                box_xyxy=[x1, y1, x2, y2],
            ))

        return results

    def detect_batch(
        self,
        images: List[Image.Image],
        classes: List[str],
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> List[List[DetectionResult]]:
        """对图片列表批量检测（逐张推理，共享模型加载开销）。"""
        return [
            self.detect(img, classes, box_threshold, text_threshold)
            for img in images
        ]


def _match_label(phrase: str, classes: List[str]) -> str:
    """将 GroundingDINO 返回的 phrase 映射回最近的类别名。"""
    phrase_lower = phrase.strip().lower()
    # 精确匹配
    for cls in classes:
        if cls.lower() == phrase_lower:
            return cls
    # 包含匹配
    for cls in classes:
        if cls.lower() in phrase_lower or phrase_lower in cls.lower():
            return cls
    # fallback
    return phrase.strip() if phrase.strip() else classes[0] if classes else "object"
