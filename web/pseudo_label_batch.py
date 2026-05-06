# -*- coding: utf-8 -*-
"""
批量伪标签：用已加载的 SAM3 对无标注图片跑英文提示词，导出 YOLO 格式 labels，供抽检后走 Web YOLO 训练。

用法（在项目根目录）:
  set SAM3_CHECKPOINT=D:\\path\\to\\sam3.pt
  python web/pseudo_label_batch.py --input D:\\raw_images --output D:\\pseudo_yolo --prompt "landslide crack"

输出目录结构（与 Web 训练兼容）:
  output/train/images  output/train/labels
  output/val/images    output/val/labels
  output/classes.txt
  output/data.yaml
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List

# 保证可从项目根以 `python web/pseudo_label_batch.py` 运行
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from PIL import Image

from web.sam_engine import _bpe_path, _ensure_sam3_on_syspath, run_text_prompt_on_state

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _default_ckpt() -> str:
    return (os.environ.get("SAM3_CHECKPOINT") or "").strip() or r"D:\BaiduNetdiskDownload\权重\sam3.pt"


def _collect_images(root: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return out


def _rects_to_yolo_lines(
    results: List[dict], img_w: int, img_h: int, class_id: int, min_score: float
) -> List[str]:
    lines: List[str] = []
    for r in results:
        if float(r.get("score", 1.0)) < min_score:
            continue
        rect = r.get("rect")
        if not rect or len(rect) < 4:
            continue
        x, y, rw, rh = rect[:4]
        if rw <= 1 or rh <= 1:
            continue
        cx = (x + rw / 2.0) / img_w
        cy = (y + rh / 2.0) / img_h
        nw = rw / img_w
        nh = rh / img_h
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        nw = max(1e-6, min(1.0, nw))
        nh = max(1e-6, min(1.0, nh))
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def _load_model(ckpt: str, confidence: float):
    _ensure_sam3_on_syspath()
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    bpe = _bpe_path()
    if not os.path.isfile(bpe):
        raise FileNotFoundError(f"BPE 不存在: {bpe}")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"权重不存在: {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(
        bpe_path=bpe,
        checkpoint_path=ckpt,
        load_from_HF=False,
        enable_inst_interactivity=True,
    )
    model.to(device)
    processor = Sam3Processor(model, confidence_threshold=confidence)
    return processor, device


def main() -> None:
    ap = argparse.ArgumentParser(description="SAM3 批量伪标签 → YOLO 目录")
    ap.add_argument("--input", type=Path, required=True, help="无标注图片文件夹（递归扫描）")
    ap.add_argument("--output", type=Path, required=True, help="输出 YOLO 数据集根目录")
    ap.add_argument("--prompt", type=str, required=True, help="英文提示词，如 landslide, crack, person")
    ap.add_argument("--class-name", type=str, default="pseudo", help="单类名，写入 classes.txt（class_id=0）")
    ap.add_argument("--checkpoint", type=str, default="", help="sam3.pt；默认读 SAM3_CHECKPOINT")
    ap.add_argument("--train-ratio", type=float, default=0.9, help="train 占比，其余进 val")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--confidence", type=float, default=0.3, help="Sam3Processor 置信度阈值")
    ap.add_argument("--min-score", type=float, default=0.0, help="额外过滤每框 score 下限")
    args = ap.parse_args()

    ckpt = args.checkpoint.strip() or _default_ckpt()
    images = _collect_images(args.input)
    if not images:
        raise SystemExit(f"未在 {args.input} 下找到图片（{IMG_EXTS}）")

    random.seed(args.seed)
    random.shuffle(images)
    n = len(images)
    if n == 1:
        # 仅一张图时 train/val 各放一份，避免后续 YOLO 训练「验证集为空」
        train_set, val_set = set(images), set(images)
    else:
        n_train = max(1, min(n - 1, int(round(n * args.train_ratio))))
        train_set = set(images[:n_train])
        val_set = set(images[n_train:])

    out = args.output
    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    (out / "classes.txt").write_text(args.class_name.strip() + "\n", encoding="utf-8")

    processor, device = _load_model(ckpt, args.confidence)
    unique = list(train_set | val_set)
    print(f"设备: {device}，权重: {ckpt}，共 {len(images)} 张图（去重推理 {len(unique)} 次）")

    def dst_name_for(src: Path) -> str:
        rel = src.relative_to(args.input)
        safe_stem = str(rel).replace("\\", "_").replace("/", "_")
        if safe_stem != src.name:
            return f"{Path(safe_stem).stem}{src.suffix.lower()}"
        return src.name

    done = 0
    for src in unique:
        dst_name = dst_name_for(src)
        pil = Image.open(src).convert("RGB")
        w, h = pil.size
        with torch.inference_mode(), torch.autocast(
            device_type="cuda" if device == "cuda" else "cpu",
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        ):
            state = processor.set_image(pil)
        results = run_text_prompt_on_state(processor, state, args.prompt)
        lines = _rects_to_yolo_lines(results, w, h, class_id=0, min_score=args.min_score)
        text = "\n".join(lines) + ("\n" if lines else "")

        for split in ("train", "val"):
            bucket = train_set if split == "train" else val_set
            if src not in bucket:
                continue
            shutil.copy2(src, out / split / "images" / dst_name)
            (out / split / "labels" / f"{Path(dst_name).stem}.txt").write_text(
                text, encoding="utf-8"
            )

        done += 1
        if done % 50 == 0 or done == len(unique):
            print(f"进度 {done}/{len(unique)}")

    data_yaml = f"""path: {out.as_posix()}
train: train/images
val: val/images
nc: 1
names:
  0: {args.class_name.strip()}
"""
    (out / "data.yaml").write_text(data_yaml, encoding="utf-8")
    print(f"完成。请人工抽检 {out}/train 与 {out}/val 的 labels，修正后再打成 zip 上传 Web 训练。")


if __name__ == "__main__":
    main()
