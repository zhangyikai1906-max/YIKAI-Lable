# -*- coding: utf-8 -*-
"""
SAM3 推理引擎 V2：在原有文本提示基础上，新增 bbox prompt 模式。

两种推理路径：
  1. text_prompt(image, text)         → 调用 SAM3 内置文本分支（原有逻辑）
  2. predict_with_boxes(image, boxes) → 用 GroundingDINO 输出的 bbox 作为提示

适配 RTX 2060 6GB：使用 bfloat16 + inference_mode 节省显存。
"""
from __future__ import annotations

import os
import sys
import threading
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _ensure_sam3_on_syspath() -> None:
    root = _project_root()
    sam3_repo = os.path.join(root, "sam3")
    if os.path.isdir(os.path.join(sam3_repo, "sam3")):
        if sam3_repo not in sys.path:
            sys.path.insert(0, sam3_repo)
    if root not in sys.path:
        sys.path.insert(0, root)


def _bpe_path() -> str:
    return os.path.join(_project_root(), "sam3", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")


class MaskResult:
    __slots__ = ("mask", "score", "poly_pts", "rect_xywh", "obb", "label")

    def __init__(
        self,
        mask: np.ndarray,
        score: float,
        poly_pts: List[List[float]],
        rect_xywh: List[float],
        obb: List[float],
        label: str = "",
    ):
        self.mask = mask             # (H, W) uint8，0/255
        self.score = score
        self.poly_pts = poly_pts     # [[x,y], ...]
        self.rect_xywh = rect_xywh  # [x, y, w, h]
        self.obb = obb               # [cx, cy, w, h, angle]
        self.label = label

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "score": self.score,
            "rect_xywh": self.rect_xywh,
            "poly_pts": self.poly_pts,
            "obb": self.obb,
        }


def _mask_tensor_to_result(
    mask_tensor: Any,
    score: float,
    label: str = "",
) -> MaskResult:
    """将单个掩膜张量转换为 MaskResult。"""
    mask_np = mask_tensor.cpu().numpy() if torch.is_tensor(mask_tensor) else mask_tensor
    mask_np = np.squeeze(mask_np)
    mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    poly_pts: List[List[float]] = []
    obb: List[float] = []

    if contours:
        largest = max(contours, key=cv2.contourArea)
        eps = 0.002 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, eps, True)
        poly_pts = approx.reshape(-1, 2).tolist()

        obox = cv2.minAreaRect(largest)
        obb = [obox[0][0], obox[0][1], obox[1][0], obox[1][1], obox[2]]

    ys, xs = np.where(mask_uint8 > 0)
    if len(xs) > 0:
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        rect_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
    else:
        rect_xywh = [0.0, 0.0, 0.0, 0.0]

    return MaskResult(
        mask=mask_uint8,
        score=score,
        poly_pts=poly_pts,
        rect_xywh=rect_xywh,
        obb=obb,
        label=label,
    )


def _empty_mask_result(
    x1: float, y1: float, x2: float, y2: float,
    label: str, H: int, W: int
) -> MaskResult:
    """创建空掩膜结果（bbox 推理失败时的占位）。"""
    mask = np.zeros((H, W), dtype=np.uint8)
    return MaskResult(
        mask=mask,
        score=0.0,
        poly_pts=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        rect_xywh=[x1, y1, x2 - x1, y2 - y1],
        obb=[],
        label=label,
    )


class SamEngineV2:
    """
    SAM3 推理引擎 V2（线程安全）。

    支持两种模式：
      - text_prompt: SAM3 内置文本分支（原有能力）
      - predict_with_boxes: 外部 bbox 提示（新增，配合 GroundingDINO 使用）
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._processor: Optional[Any] = None
        self.load_error: Optional[str] = None
        self._current_state: Optional[Dict] = None
        self._current_image: Optional[Image.Image] = None

    def is_ready(self) -> bool:
        return self._model is not None and self._processor is not None

    def load(self, checkpoint_path: str) -> Dict:
        """加载 SAM3 权重。"""
        with self._lock:
            self.load_error = None
            _ensure_sam3_on_syspath()

            try:
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor
            except Exception as e:
                self.load_error = (
                    f"无法导入 sam3（请从项目根目录启动）: {type(e).__name__}: {e}"
                )
                return {"ok": False, "message": self.load_error}

            bpe = _bpe_path()
            if not os.path.exists(bpe):
                self.load_error = f"BPE 词表缺失: {bpe}"
                return {"ok": False, "message": self.load_error}

            if not checkpoint_path or not os.path.isfile(checkpoint_path):
                self.load_error = f"SAM3 权重文件不存在: {checkpoint_path}"
                return {"ok": False, "message": self.load_error}

            try:
                model = build_sam3_image_model(
                    bpe_path=bpe,
                    checkpoint_path=checkpoint_path,
                    enable_inst_interactivity=True,
                )
                model.to("cuda" if torch.cuda.is_available() else "cpu")
                model.eval()
                self._processor = Sam3Processor(model, confidence_threshold=0.3)
                self._model = model
                self._current_state = None
                self._current_image = None
            except Exception as e:
                self._model = None
                self._processor = None
                self.load_error = str(e)
                return {"ok": False, "message": self.load_error}

            return {"ok": True, "message": "SAM3 加载成功"}

    def set_image(self, image: Image.Image) -> Dict:
        """预提取图像特征（后续推理直接复用）。"""
        with self._lock:
            if not self._processor:
                return {"ok": False, "message": "模型未加载"}
            try:
                device = next(self._model.parameters()).device
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16
                ):
                    state = self._processor.set_image(image)
                self._current_state = state
                self._current_image = image
            except Exception as e:
                self._current_state = None
                return {"ok": False, "message": str(e)}
            return {"ok": True, "width": image.width, "height": image.height}

    def text_prompt(self, prompt: str) -> List[MaskResult]:
        """
        SAM3 文本提示推理（原有能力保留）。
        必须先调用 set_image。
        """
        if not self._processor or not self._current_state:
            raise RuntimeError("请先调用 set_image")

        with self._lock:
            device = next(self._model.parameters()).device
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16
            ):
                out = self._processor.set_text_prompt(
                    prompt=prompt, state=self._current_state
                )

        masks = out.get("masks", [])
        scores = out.get("scores", [])
        results: List[MaskResult] = []
        for i in range(len(masks)):
            score = float(scores[i].cpu() if torch.is_tensor(scores[i]) else scores[i])
            results.append(_mask_tensor_to_result(masks[i], score, label=prompt))
        return results

    def predict_with_boxes(
        self,
        image: Image.Image,
        boxes_xyxy: List[List[float]],
        labels: Optional[List[str]] = None,
    ) -> List[MaskResult]:
        """
        使用外部 bbox 作为提示进行分割。

        Args:
            image:       PIL RGB 图像
            boxes_xyxy:  [[x1,y1,x2,y2], ...] 像素坐标（绝对值）
            labels:      每个 bbox 对应的类别名（可选）

        Returns:
            MaskResult 列表，与 boxes_xyxy 一一对应

        SAM3 API 说明：
          - add_geometric_prompt(box=[cx,cy,w,h] 归一化, label=True, state)
          - 每个 bbox 调用前需 reset_all_prompts 清理上一次的 geometric_prompt
          - 不可嵌套持锁调用（set_image 内部已持锁，本方法直接操作 state）
        """
        if not self.is_ready():
            raise RuntimeError("SAM3 模型未加载")

        if labels is None:
            labels = [""] * len(boxes_xyxy)

        W, H = image.width, image.height

        # 先在锁外检查是否需要 set_image
        need_set = self._current_image is not image
        if need_set:
            resp = self.set_image(image)   # set_image 内部自行加锁，不嵌套
            if not resp.get("ok"):
                raise RuntimeError(f"图像特征提取失败: {resp.get('message')}")

        results: List[MaskResult] = []

        with self._lock:
            for box_xyxy, label in zip(boxes_xyxy, labels):
                try:
                    # 每框推理前清理上一次 geometric_prompt
                    self._processor.reset_all_prompts(self._current_state)

                    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
                    # 转为 SAM3 要求的归一化 [cx, cy, w, h]
                    cx = (x1 + x2) / 2.0 / W
                    cy = (y1 + y2) / 2.0 / H
                    bw = (x2 - x1) / W
                    bh = (y2 - y1) / H

                    # SAM3 模型权重为 bfloat16，必须保持相同的 autocast 上下文
                    device_type = next(self._model.parameters()).device.type
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        state = self._processor.add_geometric_prompt(
                            box=[cx, cy, bw, bh],
                            label=True,          # positive box
                            state=self._current_state,
                        )

                    masks  = state.get("masks")   # bool tensor (N, H, W)
                    scores = state.get("scores")  # tensor (N,)

                    if masks is None or len(masks) == 0:
                        results.append(_empty_mask_result(x1, y1, x2, y2, label, H, W))
                        continue

                    best_idx = int(scores.argmax())
                    best_score = float(scores[best_idx].cpu())
                    results.append(_mask_tensor_to_result(masks[best_idx], best_score, label=label))

                except Exception as e:
                    import traceback as _tb
                    print(f"[SAM3] bbox {box_xyxy} 推理失败: {e}\n{_tb.format_exc()}", flush=True)
                    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
                    results.append(_empty_mask_result(x1, y1, x2, y2, label, H, W))

        return results

    def predict_image_bytes(
        self,
        image_bytes: bytes,
        boxes_xyxy: List[List[float]],
        labels: Optional[List[str]] = None,
    ) -> List[MaskResult]:
        """便捷接口：直接传字节数据。"""
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return self.predict_with_boxes(image, boxes_xyxy, labels)
