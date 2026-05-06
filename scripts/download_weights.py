#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键下载 LuoHuaLabel v2 所需模型权重。

下载目标（到 weights/ 目录）：
  - GroundingDINO-SwinT 配置文件  (~3 KB)
  - GroundingDINO-SwinT 权重       (~694 MB)
  - SAM2.1 Hiera-Small 权重        (~185 MB)  (可选，若已有 sam3.pt 可跳过)

用法：
    python scripts/download_weights.py
    python scripts/download_weights.py --skip-gdino   # 跳过 GroundingDINO
    python scripts/download_weights.py --skip-sam     # 跳过 SAM2.1
    python scripts/download_weights.py --weights-dir D:\\my_weights
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

WEIGHTS = {
    "gdino_config": {
        "url": (
            "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/"
            "main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        ),
        "filename": "GroundingDINO_SwinT_OGC.py",
        "size_mb": 0.01,
        "skip_flag": "skip_gdino",
    },
    "gdino_weights": {
        "url": (
            "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/"
            "groundingdino_swint_ogc.pth"
        ),
        "filename": "groundingdino_swint_ogc.pth",
        "size_mb": 694,
        "skip_flag": "skip_gdino",
    },
    "sam21_small": {
        "url": (
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
            "sam2.1_hiera_small.pt"
        ),
        "filename": "sam2.1_hiera_small.pt",
        "size_mb": 185,
        "skip_flag": "skip_sam",
    },
}


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        mb_done = downloaded / 1024 / 1024
        mb_total = total_size / 1024 / 1024
        sys.stdout.write(f"\r  [{bar}] {pct:.1f}%  {mb_done:.1f}/{mb_total:.1f} MB")
        sys.stdout.flush()
    if block_num * block_size >= total_size:
        print()


def download_file(url: str, dest: Path, size_mb: float) -> None:
    print(f"  → {dest.name}  (~{size_mb:.0f} MB)")
    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=_progress_hook)
        print(f"  ✓ 保存至: {dest}")
    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        print(f"    请手动从以下地址下载并放入 weights/ 目录：\n    {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 LuoHuaLabel v2 模型权重")
    parser.add_argument("--weights-dir", default="weights", help="权重保存目录（默认: weights/）")
    parser.add_argument("--skip-gdino", action="store_true", help="跳过 GroundingDINO 权重")
    parser.add_argument("--skip-sam", action="store_true", help="跳过 SAM2.1 权重")
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir).resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔽 LuoHuaLabel v2 权重下载")
    print(f"   保存目录: {weights_dir}\n")

    for key, info in WEIGHTS.items():
        skip_flag = info.get("skip_flag", "")
        if skip_flag == "skip_gdino" and args.skip_gdino:
            print(f"  ⏭ 跳过 {info['filename']}")
            continue
        if skip_flag == "skip_sam" and args.skip_sam:
            print(f"  ⏭ 跳过 {info['filename']}")
            continue

        dest = weights_dir / info["filename"]
        if dest.exists():
            print(f"  ✓ 已存在，跳过: {dest.name}")
            continue

        download_file(info["url"], dest, info["size_mb"])

    print("\n✅ 下载完成！")
    print("\n📋 启动服务时设置以下环境变量：")
    print(f'   set SAM3_CHECKPOINT={weights_dir}\\sam2.1_hiera_small.pt')
    print(f'   set GDINO_CONFIG={weights_dir}\\GroundingDINO_SwinT_OGC.py')
    print(f'   set GDINO_CHECKPOINT={weights_dir}\\groundingdino_swint_ogc.pth')
    print('\n🚀 启动命令：')
    print('   python -m uvicorn web.app_v2:app --host 0.0.0.0 --port 8081')
    print('\n🌐 浏览器访问：http://localhost:8081\n')


if __name__ == "__main__":
    main()
