# -*- coding: utf-8 -*-
"""
图片上传服务：支持 zip 包、多文件上传两种方式。

输出目录结构：
    label_jobs/<job_id>/images/*.jpg|png|...

支持的图片格式：jpg, jpeg, png, bmp, webp, tiff
最大 zip 大小：2 GB
"""
from __future__ import annotations

import io
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
MAX_ZIP_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


class UploadResult:
    def __init__(self, job_id: str, job_dir: Path):
        self.job_id = job_id
        self.job_dir = job_dir
        self.images_dir = job_dir / "images"
        self.image_count: int = 0
        self.skipped_count: int = 0
        self.errors: List[str] = []

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "image_count": self.image_count,
            "skipped_count": self.skipped_count,
            "errors": self.errors[:10],
        }


class UploadService:
    """
    图片上传处理服务。

    Args:
        jobs_base_dir: label_jobs/ 的父目录（通常为 web/ 目录）
    """

    def __init__(self, jobs_base_dir: Path):
        self.jobs_dir = jobs_base_dir / "label_jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def create_job(self) -> str:
        """创建新任务目录，返回 job_id。"""
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.jobs_dir / job_id
        (job_dir / "images").mkdir(parents=True, exist_ok=True)
        (job_dir / "results").mkdir(parents=True, exist_ok=True)
        (job_dir / "exports").mkdir(parents=True, exist_ok=True)
        return job_id

    def get_job_dir(self, job_id: str) -> Optional[Path]:
        d = self.jobs_dir / job_id
        return d if d.exists() else None

    def handle_zip_upload(self, job_id: str, zip_bytes: bytes) -> UploadResult:
        """
        处理 zip 包上传。
        自动识别 zip 内的图片文件（递归搜索），忽略非图片文件。
        """
        job_dir = self.jobs_dir / job_id
        result = UploadResult(job_id, job_dir)

        if len(zip_bytes) > MAX_ZIP_SIZE_BYTES:
            result.errors.append(f"zip 文件过大（>{MAX_ZIP_SIZE_BYTES // 1024 // 1024} MB）")
            return result

        images_dir = job_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue

                    member_path = Path(member.filename)
                    if member_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                        result.skipped_count += 1
                        continue

                    # 扁平化：只保留文件名（去掉子目录路径，避免冲突则加前缀）
                    dest_name = _safe_filename(member_path.name, images_dir)
                    dest_path = images_dir / dest_name

                    try:
                        data = zf.read(member.filename)
                        # 验证是合法图片
                        img = Image.open(io.BytesIO(data))
                        img.verify()
                        dest_path.write_bytes(data)
                        result.image_count += 1
                    except Exception as e:
                        result.skipped_count += 1
                        result.errors.append(f"{member.filename}: {e}")

        except zipfile.BadZipFile:
            result.errors.append("无效的 zip 文件")
        except Exception as e:
            result.errors.append(f"解压失败: {e}")

        return result

    def handle_files_upload(self, job_id: str, files: List[Tuple[str, bytes]]) -> UploadResult:
        """
        处理多文件上传。

        Args:
            files: [(filename, file_bytes), ...]
        """
        job_dir = self.jobs_dir / job_id
        result = UploadResult(job_id, job_dir)
        images_dir = job_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        for filename, data in files:
            suffix = Path(filename).suffix.lower()
            if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
                result.skipped_count += 1
                continue

            dest_name = _safe_filename(filename, images_dir)
            dest_path = images_dir / dest_name

            try:
                img = Image.open(io.BytesIO(data))
                img.verify()
                dest_path.write_bytes(data)
                result.image_count += 1
            except Exception as e:
                result.skipped_count += 1
                result.errors.append(f"{filename}: {e}")

        return result

    def list_images(self, job_id: str) -> List[Path]:
        """返回任务的图片路径列表（已排序）。"""
        job_dir = self.jobs_dir / job_id
        images_dir = job_dir / "images"
        if not images_dir.exists():
            return []
        paths = [
            p for p in sorted(images_dir.iterdir())
            if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        return paths

    def delete_job(self, job_id: str) -> bool:
        """删除任务目录（释放磁盘空间）。"""
        job_dir = self.jobs_dir / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir)
            return True
        return False

    def job_stats(self, job_id: str) -> dict:
        """返回任务的磁盘占用统计。"""
        job_dir = self.jobs_dir / job_id
        if not job_dir.exists():
            return {}

        images_count = len(self.list_images(job_id))
        results_count = len(list((job_dir / "results").glob("*.json"))) if (job_dir / "results").exists() else 0

        total_size = sum(f.stat().st_size for f in job_dir.rglob("*") if f.is_file())

        return {
            "job_id": job_id,
            "images": images_count,
            "labeled": results_count,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
        }


def _safe_filename(filename: str, dest_dir: Path) -> str:
    """
    若目标目录已存在同名文件，自动在文件名前加数字前缀避免冲突。
    """
    name = Path(filename).name
    if not (dest_dir / name).exists():
        return name

    stem = Path(name).stem
    suffix = Path(name).suffix
    i = 1
    while True:
        new_name = f"{stem}_{i}{suffix}"
        if not (dest_dir / new_name).exists():
            return new_name
        i += 1
