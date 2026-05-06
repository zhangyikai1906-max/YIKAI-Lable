# -*- coding: utf-8 -*-
"""
无 Qt 依赖的 SAM3 推理封装，供 Web API 使用。
逻辑与 core/sam_client.py 中文本分支一致。
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_sam3_on_syspath() -> None:
    """把本地 sam3 源码目录加入 sys.path，避免从非项目根启动 uvicorn 时 import 失败。"""
    root = _project_root()
    sam3_repo = os.path.join(root, "sam3")
    inner_pkg = os.path.join(sam3_repo, "sam3")
    if os.path.isdir(inner_pkg):
        if sam3_repo not in sys.path:
            sys.path.insert(0, sam3_repo)
    if root not in sys.path:
        sys.path.insert(0, root)


def _bpe_path() -> str:
    return os.path.join(_project_root(), "sam3", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")


def run_text_prompt_on_state(
    processor: Any,
    inference_state: Dict[str, Any],
    prompt_text: str,
) -> List[Dict[str, Any]]:
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out_state = processor.set_text_prompt(prompt=prompt_text, state=inference_state)

    masks = out_state.get("masks", [])
    scores = out_state.get("scores", [])
    boxes = out_state.get("boxes", [])

    results: List[Dict[str, Any]] = []
    if len(masks) == 0:
        return results

    for i in range(len(masks)):
        mask_np = masks[i].cpu().numpy() if torch.is_tensor(masks[i]) else masks[i]
        mask_np = np.squeeze(mask_np)

        score_val = float(scores[i].cpu() if torch.is_tensor(scores[i]) else scores[i])
        box = boxes[i].cpu().numpy() if torch.is_tensor(boxes[i]) else boxes[i]

        if box.ndim > 1:
            box = box.squeeze()
        x1, y1, x2, y2 = box
        rect_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

        mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        poly_pts: List[List[float]] = []
        rect_obb: List[float] = []
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.002 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            poly_pts = approx.reshape(-1, 2).tolist()

            obb = cv2.minAreaRect(largest_contour)
            rect_obb = [obb[0][0], obb[0][1], obb[1][0], obb[1][1], obb[2]]

        if poly_pts or rect_xywh:
            results.append({
                "poly_pts": poly_pts,
                "rect": rect_xywh,
                "obb": rect_obb,
                "score": score_val,
            })
    return results


class SamWebEngine:
    """单例式引擎：加载一次模型，按请求处理图片与文本提示。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.model = None
        self.processor: Optional[Any] = None
        self.inference_state: Optional[Dict[str, Any]] = None
        self.load_error: Optional[str] = None
        self.image_width = 0
        self.image_height = 0

    def is_ready(self) -> bool:
        return self.model is not None and self.processor is not None

    def load_model(self, checkpoint_path: str) -> Dict[str, Any]:
        with self._lock:
            self.load_error = None
            _ensure_sam3_on_syspath()
            try:
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor
            except Exception as e:
                self.load_error = (
                    f"无法导入 sam3（请确认项目内存在 sam3/sam3 源码，并从项目根目录启动服务）: "
                    f"{type(e).__name__}: {e}"
                )
                return {"ok": False, "message": self.load_error}

            bpe = _bpe_path()
            if not os.path.exists(bpe):
                self.load_error = f"未找到 BPE 词表: {bpe}"
                return {"ok": False, "message": self.load_error}

            if not checkpoint_path or not os.path.isfile(checkpoint_path):
                self.load_error = f"权重文件不存在: {checkpoint_path}"
                return {"ok": False, "message": self.load_error}

            try:
                model = build_sam3_image_model(
                    bpe_path=bpe,
                    checkpoint_path=checkpoint_path,
                    enable_inst_interactivity=True,
                )
                model.to("cuda")
                self.processor = Sam3Processor(model, confidence_threshold=0.3)
                self.model = model
                self.inference_state = None
            except Exception as e:
                self.model = None
                self.processor = None
                self.load_error = str(e)
                return {"ok": False, "message": self.load_error}

            return {"ok": True, "message": "模型加载成功"}

    def set_image(self, image_bytes: bytes) -> Dict[str, Any]:
        with self._lock:
            if not self.processor:
                return {"ok": False, "message": "模型未加载"}

            try:
                from io import BytesIO

                pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
                self.image_width, self.image_height = pil_img.size

                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    state = self.processor.set_image(pil_img)
                self.inference_state = state
            except Exception as e:
                self.inference_state = None
                return {"ok": False, "message": str(e)}

            return {
                "ok": True,
                "message": "图片已就绪",
                "width": self.image_width,
                "height": self.image_height,
            }

    def text_prompt(self, prompt: str) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        if not prompt:
            return {"ok": False, "message": "提示词为空", "results": []}

        with self._lock:
            if not self.processor or not self.inference_state:
                return {"ok": False, "message": "请先上传图片并完成特征提取", "results": []}

            try:
                results = run_text_prompt_on_state(
                    self.processor, self.inference_state, prompt
                )
            except Exception as e:
                return {"ok": False, "message": str(e), "results": []}

            return {"ok": True, "message": "ok", "results": results, "prompt": prompt}
