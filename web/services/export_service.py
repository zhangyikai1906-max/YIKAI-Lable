# -*- coding: utf-8 -*-
"""
多格式数据集导出服务。

支持格式：
  - YOLO      images/ + labels/*.txt + data.yaml
  - COCO      images/ + _annotations.coco.json
  - VOC       images/ + Annotations/*.xml (Pascal VOC)
  - LabelMe   images/ + labels_labelme/*.json
  - Masks     images/ + masks/*.png (二值掩膜)

输入：label_jobs/<job_id>/results/*.json（ImageAnnotation 格式）
输出：label_jobs/<job_id>/exports/<format>/
导出包：label_jobs/<job_id>/exports/<format>.zip
"""
from __future__ import annotations

import io
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw


SUPPORTED_FORMATS = ["yolo", "coco", "voc", "labelme", "masks"]


# ─────────────────────────────────────────────
# 主导出函数
# ─────────────────────────────────────────────

class ExportService:
    """
    数据集导出服务。

    使用方式：
        svc = ExportService(job_dir)
        zip_path = svc.export("yolo")
    """

    def __init__(self, job_dir: Path):
        """
        Args:
            job_dir: label_jobs/<job_id>/ 目录，包含：
                     images/   原始图片
                     results/  每张图的 ImageAnnotation JSON
        """
        self.job_dir = job_dir
        self.images_dir = job_dir / "images"
        self.results_dir = job_dir / "results"
        self.exports_dir = job_dir / "exports"

    def export(
        self,
        fmt: str,
        classes: Optional[List[str]] = None,
        split_ratio: float = 0.8,
    ) -> Path:
        """
        执行导出，返回 zip 文件路径。

        Args:
            fmt:         格式名称，见 SUPPORTED_FORMATS
            classes:     类别列表（若 None，从结果中自动推断）
            split_ratio: train/val 划分比例（YOLO/COCO/VOC 生效）

        Returns:
            zip 文件的绝对路径
        """
        fmt = fmt.lower().strip()
        if fmt not in SUPPORTED_FORMATS:
            raise ValueError(f"不支持的格式: {fmt}，可选: {SUPPORTED_FORMATS}")

        annotations = self._load_all_annotations()
        if classes is None:
            classes = _infer_classes(annotations)

        out_dir = self.exports_dir / fmt
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if fmt == "yolo":
            self._export_yolo(annotations, classes, out_dir, split_ratio)
        elif fmt == "coco":
            self._export_coco(annotations, classes, out_dir, split_ratio)
        elif fmt == "voc":
            self._export_voc(annotations, classes, out_dir, split_ratio)
        elif fmt == "labelme":
            self._export_labelme(annotations, out_dir)
        elif fmt == "masks":
            self._export_masks(annotations, out_dir)

        zip_path = self.exports_dir / f"{fmt}.zip"
        _zip_directory(out_dir, zip_path)
        return zip_path

    # ─────────────────────────────────────────
    # YOLO 格式
    # ─────────────────────────────────────────

    def _export_yolo(
        self,
        annotations: List[Dict],
        classes: List[str],
        out_dir: Path,
        split_ratio: float,
    ) -> None:
        """
        YOLO 格式：
            train/images/*.jpg
            train/labels/*.txt
            val/images/*.jpg
            val/labels/*.txt
            data.yaml
        """
        class_to_id = {c: i for i, c in enumerate(classes)}
        train_ann, val_ann = _split_dataset(annotations, split_ratio)

        for split_name, split_ann in [("train", train_ann), ("val", val_ann)]:
            img_dir = out_dir / split_name / "images"
            lbl_dir = out_dir / split_name / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            for ann in split_ann:
                src_img = self.images_dir / ann["image"]
                if src_img.exists():
                    shutil.copy2(src_img, img_dir / ann["image"])

                w, h = ann["width"], ann["height"]
                lines = []
                for obj in ann.get("annotations", []):
                    label = obj["label"]
                    cls_id = class_to_id.get(label, 0)
                    x, y, bw, bh = obj["box_xywh"]
                    # 转为归一化 cx,cy,w,h
                    cx = (x + bw / 2) / w
                    cy = (y + bh / 2) / h
                    nw = bw / w
                    nh = bh / h
                    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

                lbl_path = lbl_dir / (Path(ann["image"]).stem + ".txt")
                lbl_path.write_text("\n".join(lines), encoding="utf-8")

        # data.yaml
        yaml_content = (
            f"path: .\n"
            f"train: train/images\n"
            f"val: val/images\n"
            f"nc: {len(classes)}\n"
            f"names: {json.dumps(classes, ensure_ascii=False)}\n"
        )
        (out_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")

    # ─────────────────────────────────────────
    # COCO 格式
    # ─────────────────────────────────────────

    def _export_coco(
        self,
        annotations: List[Dict],
        classes: List[str],
        out_dir: Path,
        split_ratio: float,
    ) -> None:
        """
        COCO 格式：
            train/images/*.jpg  + train/_annotations.coco.json
            val/images/*.jpg    + val/_annotations.coco.json
        """
        class_to_id = {c: i + 1 for i, c in enumerate(classes)}
        coco_categories = [
            {"id": i + 1, "name": c, "supercategory": "object"}
            for i, c in enumerate(classes)
        ]
        train_ann, val_ann = _split_dataset(annotations, split_ratio)

        for split_name, split_ann in [("train", train_ann), ("val", val_ann)]:
            img_dir = out_dir / split_name / "images"
            img_dir.mkdir(parents=True, exist_ok=True)

            coco_images = []
            coco_annotations = []
            ann_id = 1

            for img_id, ann in enumerate(split_ann, start=1):
                src_img = self.images_dir / ann["image"]
                if src_img.exists():
                    shutil.copy2(src_img, img_dir / ann["image"])

                coco_images.append({
                    "id": img_id,
                    "file_name": ann["image"],
                    "width": ann["width"],
                    "height": ann["height"],
                })

                for obj in ann.get("annotations", []):
                    label = obj["label"]
                    cat_id = class_to_id.get(label, 1)
                    x, y, bw, bh = obj["box_xywh"]
                    area = bw * bh

                    # 多边形分割
                    segmentation = []
                    if obj.get("poly_pts"):
                        flat = [coord for pt in obj["poly_pts"] for coord in pt]
                        if len(flat) >= 6:
                            segmentation = [flat]

                    coco_annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                        "area": round(area, 2),
                        "segmentation": segmentation,
                        "iscrowd": 0,
                    })
                    ann_id += 1

            coco_json = {
                "info": {"description": "Generated by LuoHuaLabel", "version": "2.0"},
                "licenses": [],
                "categories": coco_categories,
                "images": coco_images,
                "annotations": coco_annotations,
            }
            coco_out = out_dir / split_name / "_annotations.coco.json"
            coco_out.write_text(
                json.dumps(coco_json, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    # ─────────────────────────────────────────
    # Pascal VOC 格式
    # ─────────────────────────────────────────

    def _export_voc(
        self,
        annotations: List[Dict],
        classes: List[str],
        out_dir: Path,
        split_ratio: float,
    ) -> None:
        """
        VOC 格式：
            images/*.jpg
            Annotations/*.xml
            ImageSets/Main/train.txt  val.txt
        """
        img_dir = out_dir / "images"
        ann_dir = out_dir / "Annotations"
        sets_dir = out_dir / "ImageSets" / "Main"
        img_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        sets_dir.mkdir(parents=True, exist_ok=True)

        train_ann, val_ann = _split_dataset(annotations, split_ratio)

        for split_name, split_ann in [("train", train_ann), ("val", val_ann)]:
            names = []
            for ann in split_ann:
                stem = Path(ann["image"]).stem
                names.append(stem)

                src_img = self.images_dir / ann["image"]
                if src_img.exists():
                    shutil.copy2(src_img, img_dir / ann["image"])

                root = ET.Element("annotation")
                ET.SubElement(root, "folder").text = "images"
                ET.SubElement(root, "filename").text = ann["image"]
                size_el = ET.SubElement(root, "size")
                ET.SubElement(size_el, "width").text = str(ann["width"])
                ET.SubElement(size_el, "height").text = str(ann["height"])
                ET.SubElement(size_el, "depth").text = "3"

                for obj in ann.get("annotations", []):
                    obj_el = ET.SubElement(root, "object")
                    ET.SubElement(obj_el, "name").text = obj["label"]
                    ET.SubElement(obj_el, "pose").text = "Unspecified"
                    ET.SubElement(obj_el, "truncated").text = "0"
                    ET.SubElement(obj_el, "difficult").text = "0"
                    bndbox = ET.SubElement(obj_el, "bndbox")
                    x1, y1, x2, y2 = obj["box_xyxy"]
                    ET.SubElement(bndbox, "xmin").text = str(int(x1))
                    ET.SubElement(bndbox, "ymin").text = str(int(y1))
                    ET.SubElement(bndbox, "xmax").text = str(int(x2))
                    ET.SubElement(bndbox, "ymax").text = str(int(y2))

                tree = ET.ElementTree(root)
                ET.indent(tree, space="  ")
                tree.write(
                    str(ann_dir / f"{stem}.xml"),
                    encoding="utf-8",
                    xml_declaration=True,
                )

            (sets_dir / f"{split_name}.txt").write_text(
                "\n".join(names), encoding="utf-8"
            )

    # ─────────────────────────────────────────
    # LabelMe JSON 格式
    # ─────────────────────────────────────────

    def _export_labelme(self, annotations: List[Dict], out_dir: Path) -> None:
        """
        每张图输出一个 LabelMe 格式 JSON，可在 LabelMe 工具中继续人工修正。
        """
        img_dir = out_dir / "images"
        lbl_dir = out_dir / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for ann in annotations:
            src_img = self.images_dir / ann["image"]
            if src_img.exists():
                shutil.copy2(src_img, img_dir / ann["image"])

            shapes = []
            for obj in ann.get("annotations", []):
                poly = obj.get("poly_pts", [])
                if poly and len(poly) >= 3:
                    shapes.append({
                        "label": obj["label"],
                        "points": poly,
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {},
                    })
                else:
                    x, y, bw, bh = obj["box_xywh"]
                    shapes.append({
                        "label": obj["label"],
                        "points": [[x, y], [x + bw, y + bh]],
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {},
                    })

            labelme_json = {
                "version": "5.0.1",
                "flags": {},
                "shapes": shapes,
                "imagePath": f"../images/{ann['image']}",
                "imageData": None,
                "imageHeight": ann["height"],
                "imageWidth": ann["width"],
            }
            out_path = lbl_dir / (Path(ann["image"]).stem + ".json")
            out_path.write_text(
                json.dumps(labelme_json, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    # ─────────────────────────────────────────
    # 二值掩膜格式
    # ─────────────────────────────────────────

    def _export_masks(self, annotations: List[Dict], out_dir: Path) -> None:
        """
        从多边形重建二值掩膜并保存为 PNG。
        目录结构：images/*.jpg  masks/*.png
        """
        img_dir = out_dir / "images"
        mask_dir = out_dir / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for ann in annotations:
            src_img = self.images_dir / ann["image"]
            if src_img.exists():
                shutil.copy2(src_img, img_dir / ann["image"])

            w, h = ann["width"], ann["height"]
            mask = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(mask)

            for obj in ann.get("annotations", []):
                poly = obj.get("poly_pts", [])
                if poly and len(poly) >= 3:
                    flat = [(float(p[0]), float(p[1])) for p in poly]
                    draw.polygon(flat, fill=255)
                else:
                    x, y, bw, bh = obj["box_xywh"]
                    draw.rectangle([x, y, x + bw, y + bh], fill=255)

            mask.save(mask_dir / (Path(ann["image"]).stem + ".png"))

    # ─────────────────────────────────────────
    # 辅助
    # ─────────────────────────────────────────

    def _load_all_annotations(self) -> List[Dict]:
        if not self.results_dir.exists():
            return []
        anns = []
        for f in sorted(self.results_dir.glob("*.json")):
            if f.name.startswith("_"):
                continue
            try:
                anns.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass
        return anns

    def list_exports(self) -> List[Dict]:
        """列出已生成的导出包信息。"""
        if not self.exports_dir.exists():
            return []
        result = []
        for fmt in SUPPORTED_FORMATS:
            zip_path = self.exports_dir / f"{fmt}.zip"
            if zip_path.exists():
                result.append({
                    "format": fmt,
                    "path": str(zip_path),
                    "size_mb": round(zip_path.stat().st_size / 1024 / 1024, 2),
                })
        return result


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def _infer_classes(annotations: List[Dict]) -> List[str]:
    """从标注结果中推断类别列表（保持首次出现顺序）。"""
    seen: Dict[str, int] = {}
    for ann in annotations:
        for obj in ann.get("annotations", []):
            label = obj.get("label", "")
            if label and label not in seen:
                seen[label] = len(seen)
    return sorted(seen.keys(), key=lambda x: seen[x])


def _split_dataset(
    annotations: List[Dict],
    train_ratio: float,
) -> tuple:
    """按比例划分 train/val（不打乱顺序，保持确定性）。"""
    n = len(annotations)
    split = max(1, int(n * train_ratio))
    return annotations[:split], annotations[split:]


def _zip_directory(src_dir: Path, zip_path: Path) -> None:
    """将目录打包为 zip。"""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in src_dir.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(src_dir))
