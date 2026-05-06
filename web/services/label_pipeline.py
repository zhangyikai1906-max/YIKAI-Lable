# -*- coding: utf-8 -*-
"""
DINOv3 × SAM3 批量标注流水线。

流程：
  1. （可选）DINOv3 特征聚类 → 代表图预览
  2. DINOv3 注意力图检测 → bbox 区域提议
     - attention 模式：无监督检测所有显著目标
     - patch_sim 模式：提供参考图，少样本匹配
  3. SAM3 bbox prompt 精分割 → 像素级掩膜
  4. NMS + 置信度过滤
  5. 结果写入 label_jobs/<job_id>/results/<stem>.json
  6. 实时回调进度

适配 RTX 2060 6GB：
  - DINOv3-vits16 (~82MB) + SAM3 (~300MB) 合计约 1.5GB
  - batch_size=4 时峰值约 2.5GB，余量充足
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from web.models.dinov3_engine import DINOv3Engine, DetectionResult
from web.models.sam_engine_v2 import SamEngineV2, MaskResult


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

class ImageAnnotation:
    def __init__(self, image_path: Path, width: int, height: int):
        self.image_path = image_path
        self.width = width
        self.height = height
        self.annotations: List[Dict] = []
        self.error: Optional[str] = None
        self.elapsed_ms: float = 0.0

    def add(self, det: DetectionResult, mask: MaskResult) -> None:
        self.annotations.append({
            "label": det.label or mask.label,
            "score": round(det.score, 4),
            "box_xyxy": [round(v, 2) for v in det.box_xyxy],
            "box_xywh": [round(v, 2) for v in det.box_xywh],
            "mask_score": round(mask.score, 4),
            "rect_xywh": [round(v, 2) for v in mask.rect_xywh],
            "poly_pts": [[round(p[0], 2), round(p[1], 2)] for p in mask.poly_pts],
            "obb": [round(v, 4) for v in mask.obb],
        })

    def to_dict(self) -> dict:
        return {
            "image": self.image_path.name,
            "width": self.width,
            "height": self.height,
            "annotations": self.annotations,
            "error": self.error,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


class PipelineConfig:
    """流水线配置参数。"""

    def __init__(
        self,
        class_names: List[str],
        # DINOv3 检测参数
        detect_mode: str = "attention",        # "attention" | "patch_sim"
        attn_threshold: float = 0.40,          # 注意力阈值（0~1）
        head_fusion: str = "mean",             # "mean" | "max" | "max_ent"
        min_area_ratio: float = 0.002,         # 最小目标面积比
        max_area_ratio: float = 0.95,          # 最大目标面积比（过滤背景）
        max_detections: int = 20,              # 每图最多检测框数
        # SAM3 分割参数
        mask_score_threshold: float = 0.5,
        nms_iou_threshold: float = 0.5,
        # 批处理
        batch_size: int = 4,
        # DINOv3 聚类（可选）
        use_dinov3_cluster: bool = False,
        dinov3_n_clusters: int = 10,
        # 少样本参考图（patch_sim 模式）
        reference_images: Optional[List[Image.Image]] = None,
    ):
        self.class_names = class_names
        self.detect_mode = detect_mode
        self.attn_threshold = attn_threshold
        self.head_fusion = head_fusion
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.max_detections = max_detections
        self.mask_score_threshold = mask_score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.batch_size = batch_size
        self.use_dinov3_cluster = use_dinov3_cluster
        self.dinov3_n_clusters = dinov3_n_clusters
        self.reference_images = reference_images or []


# ─────────────────────────────────────────────
# NMS
# ─────────────────────────────────────────────

def _iou(a: List[float], b: List[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _nms(dets: List[DetectionResult], iou_thr: float) -> List[DetectionResult]:
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    keep: List[DetectionResult] = []
    for d in dets:
        if all(_iou(d.box_xyxy, k.box_xyxy) <= iou_thr for k in keep):
            keep.append(d)
    return keep


# ─────────────────────────────────────────────
# 主流水线
# ─────────────────────────────────────────────

class LabelPipeline:
    """
    DINOv3 × SAM3 批量标注流水线。

    使用方式：
        pipeline = LabelPipeline(dinov3_engine, sam_engine)
        pipeline.run(image_paths, output_dir, config, progress_callback)
    """

    def __init__(self, dinov3: DINOv3Engine, sam: SamEngineV2):
        self._dinov3 = dinov3
        self._sam = sam

    def run(
        self,
        image_paths: List[Path],
        output_dir: Path,
        config: PipelineConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[ImageAnnotation]:
        output_dir.mkdir(parents=True, exist_ok=True)
        total = len(image_paths)
        results: List[ImageAnnotation] = []

        def _cb(done: int, msg: str = "") -> None:
            if progress_callback:
                progress_callback(done, total, msg)

        # ── 可选：DINOv3 聚类，输出代表图索引
        cluster_labels: Optional[np.ndarray] = None
        representative_idx: List[int] = []
        if config.use_dinov3_cluster and self._dinov3 and self._dinov3.is_ready():
            _cb(0, "DINOv3 特征提取中...")
            images_pil = [Image.open(p).convert("RGB") for p in image_paths]
            features = self._dinov3.extract(images_pil, batch_size=32)
            n_clusters = min(config.dinov3_n_clusters, total)
            cluster_labels = self._dinov3.cluster(features, n_clusters=n_clusters)
            representative_idx = self._dinov3.representative_indices(features, cluster_labels)
            _cb(0, f"DINOv3 聚类完成，{n_clusters} 个簇，{len(representative_idx)} 张代表图")

        # ── 主循环
        for img_path in image_paths:
            ann = self._process_single(img_path, config, output_dir)
            results.append(ann)
            _cb(
                len(results),
                f"[{len(results)}/{total}] {img_path.name}"
                + (f" — {ann.error}" if ann.error else f" — {len(ann.annotations)} 目标"),
            )

        # 写入聚类元数据
        if cluster_labels is not None:
            meta = {
                "cluster_labels": cluster_labels.tolist(),
                "representative_indices": representative_idx,
                "image_names": [p.name for p in image_paths],
            }
            (output_dir / "_cluster_meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        return results

    def _process_single(
        self,
        img_path: Path,
        config: PipelineConfig,
        output_dir: Path,
    ) -> ImageAnnotation:
        t0 = time.perf_counter()

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            ann = ImageAnnotation(img_path, 0, 0)
            ann.error = f"图片读取失败: {e}"
            return ann

        ann = ImageAnnotation(img_path, image.width, image.height)

        try:
            # ── Step 1: DINOv3 检测（注意力 or 相似度）
            detector = self._dinov3.detector
            detections = detector.detect(
                image=image,
                class_names=config.class_names,
                attn_threshold=config.attn_threshold,
                min_area_ratio=config.min_area_ratio,
                max_area_ratio=config.max_area_ratio,
                max_detections=config.max_detections,
                head_fusion=config.head_fusion,
                detect_mode=config.detect_mode,
                reference_images=config.reference_images if config.detect_mode == "patch_sim" else None,
            )

            # ── Step 2: NMS
            detections = _nms(detections, config.nms_iou_threshold)

            if not detections:
                ann.elapsed_ms = (time.perf_counter() - t0) * 1000
                _write_json(ann, output_dir)
                return ann

            # ── Step 3: SAM3 bbox prompt 精分割
            boxes = [d.box_xyxy for d in detections]
            labels = [d.label for d in detections]
            mask_results = self._sam.predict_with_boxes(image, boxes, labels)

            # ── Step 4: 过滤低置信度掩膜
            for det, mask in zip(detections, mask_results):
                if mask.score >= config.mask_score_threshold or mask.score == 0.0:
                    ann.add(det, mask)

        except Exception as e:
            ann.error = str(e)

        ann.elapsed_ms = (time.perf_counter() - t0) * 1000
        _write_json(ann, output_dir)
        return ann


def _write_json(ann: ImageAnnotation, output_dir: Path) -> None:
    (output_dir / f"{ann.image_path.stem}.json").write_text(
        json.dumps(ann.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
