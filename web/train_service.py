# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import importlib.util
import os
import random
import shutil
import subprocess
import threading
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _to_posix(path: Path) -> str:
    return path.as_posix()


@dataclass
class TrainJob:
    job_id: str
    status: str = "pending"
    message: str = ""
    log_lines: List[str] = field(default_factory=list)
    work_dir: Optional[Path] = None
    process: Optional[subprocess.Popen] = None
    output_ckpt: Optional[str] = None

    def append_log(self, line: str) -> None:
        line = line.rstrip("\n")
        if not line:
            return
        self.log_lines.append(line)
        if len(self.log_lines) > 1000:
            self.log_lines = self.log_lines[-1000:]


class TrainService:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.jobs_root = project_root / "web" / "train_jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, TrainJob] = {}
        self._lock = threading.Lock()

    def get_job(self, job_id: str) -> Optional[TrainJob]:
        return self.jobs.get(job_id)

    def start_job_from_yolo_zip(
        self,
        zip_bytes: bytes,
        epochs: int = 3,
        train_ratio: float = 0.8,
    ) -> TrainJob:
        job_id = uuid.uuid4().hex[:8]
        job = TrainJob(job_id=job_id, status="preparing", message="正在准备数据集")
        work_dir = self.jobs_root / job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        job.work_dir = work_dir
        self.jobs[job_id] = job

        t = threading.Thread(
            target=self._prepare_and_run,
            args=(job, zip_bytes, epochs, train_ratio),
            daemon=True,
        )
        t.start()
        return job

    def _prepare_and_run(
        self,
        job: TrainJob,
        zip_bytes: bytes,
        epochs: int,
        train_ratio: float,
    ) -> None:
        try:
            zip_path = job.work_dir / "dataset.zip"
            zip_path.write_bytes(zip_bytes)
            extract_dir = job.work_dir / "raw"
            extract_dir.mkdir(exist_ok=True)

            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

            job.append_log("已解压 YOLO zip")
            yolo_root = self._find_yolo_dataset_root(extract_dir)
            classes = self._load_classes(yolo_root)
            if not classes:
                raise ValueError("未找到类别信息（请在 zip 中提供 classes.txt 或 data.yaml names）")
            job.append_log(f"类别: {classes}")

            train_pairs, val_pairs, mode = self._resolve_train_val_pairs(
                yolo_root, train_ratio
            )
            if len(train_pairs) < 3:
                raise ValueError("训练集有效图片-标签对不足（至少 3 对）")
            if len(val_pairs) < 1:
                raise ValueError("验证集为空，请提供 val 划分或更多样本")
            job.append_log(
                f"划分方式: {mode}；train={len(train_pairs)}, val={len(val_pairs)}"
            )

            ds_root = job.work_dir / "dataset" / "custom"
            train_dir = ds_root / "train"
            val_dir = ds_root / "test"
            train_dir.mkdir(parents=True, exist_ok=True)
            val_dir.mkdir(parents=True, exist_ok=True)

            train_coco = self._build_split(train_pairs, train_dir, classes)
            val_coco = self._build_split(val_pairs, val_dir, classes)
            (train_dir / "_annotations.coco.json").write_text(
                json.dumps(train_coco, ensure_ascii=False), encoding="utf-8"
            )
            (val_dir / "_annotations.coco.json").write_text(
                json.dumps(val_coco, ensure_ascii=False), encoding="utf-8"
            )
            job.append_log("已转换为 COCO 标注")

            cfg_path = self._build_train_config(job, ds_root.parent, epochs)
            job.append_log(f"训练配置: {cfg_path}")
            self._check_train_runtime_requirements()

            self._run_training_subprocess(job, cfg_path)
        except Exception as e:
            job.status = "failed"
            job.message = f"准备失败: {e}"
            job.append_log(job.message)

    def _check_train_runtime_requirements(self) -> None:
        """在启动训练前检查关键依赖，避免子进程长堆栈。"""
        missing: List[str] = []
        for mod in ("tensorboard", "fvcore"):
            if importlib.util.find_spec(mod) is None:
                missing.append(mod)
        if missing:
            raise RuntimeError(
                "训练环境缺少依赖: "
                + ", ".join(missing)
                + "；请先执行 `pip install -r requirements-web.txt`（或单独 pip install "
                + " ".join(missing)
                + "）"
            )

    def _resolve_sam3_checkpoint(self) -> Path:
        """与 Web 推理一致：优先 SAM3_CHECKPOINT；离线环境不可从 HuggingFace 下载。"""
        raw = (os.environ.get("SAM3_CHECKPOINT") or "").strip()
        if not raw:
            raw = r"D:\BaiduNetdiskDownload\权重\sam3.pt"
        p = Path(raw)
        if not p.is_file():
            raise RuntimeError(
                f"未找到本地 SAM3 权重: {p}。请设置环境变量 SAM3_CHECKPOINT 为 sam3.pt 的绝对路径，"
                "并确保文件存在（内网/SSL 异常时无法自动从 HuggingFace 下载）。"
            )
        return p.resolve()

    def _run_training_subprocess(self, job: TrainJob, cfg_path: Path) -> None:
        job.status = "running"
        job.message = "训练中"
        cmd = [
            "python",
            "-m",
            "sam3.train.train",
            "-c",
            f"configs/web_jobs/{cfg_path.name}",
            "--use-cluster",
            "0",
            "--num-gpus",
            "1",
        ]
        env = os.environ.copy()
        py_path = f"{self.project_root / 'sam3'};{self.project_root}"
        env["PYTHONPATH"] = f"{py_path};{env.get('PYTHONPATH', '')}"
        # 某些 Windows 版 PyTorch 未编译 libuv，需显式关闭。
        env.setdefault("USE_LIBUV", "0")
        proc = subprocess.Popen(
            cmd,
            cwd=str(self.project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        job.process = proc
        assert proc.stdout is not None
        for line in proc.stdout:
            job.append_log(line)

        code = proc.wait()
        if code == 0:
            job.status = "done"
            job.message = "训练完成"
            ckpt_dir = job.work_dir / "logs" / "checkpoints"
            if ckpt_dir.exists():
                cands = sorted(ckpt_dir.glob("*.pt"))
                if cands:
                    job.output_ckpt = str(cands[-1])
        else:
            job.status = "failed"
            job.message = f"训练失败，退出码 {code}"

    def _build_train_config(self, job: TrainJob, dataset_root: Path, epochs: int) -> Path:
        src = (
            self.project_root
            / "sam3"
            / "sam3"
            / "train"
            / "configs"
            / "roboflow_v100"
            / "roboflow_v100_full_ft_100_images.yaml"
        )
        txt = src.read_text(encoding="utf-8")
        bpe = self.project_root / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        exp = job.work_dir / "logs"
        txt = txt.replace(
            "  bpe_path: <BPE_PATH> # This should be under sam3/assets/bpe_simple_vocab_16e6.txt.gz\n",
            "  bpe_path: <BPE_PATH> # This should be under sam3/assets/bpe_simple_vocab_16e6.txt.gz\n"
            "  sam3_checkpoint: <SAM3_CHECKPOINT_PATH>\n",
        )
        ckpt = self._resolve_sam3_checkpoint()
        job.append_log(f"使用本地 SAM3 权重: {ckpt}")
        txt = txt.replace("<SAM3_CHECKPOINT_PATH>", _to_posix(ckpt))
        txt = txt.replace("<YOUR_DATASET_DIR>", _to_posix(dataset_root))
        txt = txt.replace("<YOUR EXPERIMENET LOG_DIR>", _to_posix(exp))
        txt = txt.replace("<BPE_PATH>", _to_posix(bpe))
        txt = txt.replace(
            "    enable_segmentation: ${scratch.enable_segmentation} # Warning: Enable this if using segmentation.\n\n  meters:",
            "    enable_segmentation: ${scratch.enable_segmentation} # Warning: Enable this if using segmentation.\n"
            "    checkpoint_path: ${paths.sam3_checkpoint}\n"
            "    load_from_HF: false\n\n  meters:",
            1,
        )
        # 必须在同一份 YAML 内就地改键：不能再追加 roboflow_train / launcher / submitit / trainer，
        # 否则会出现重复键，OmegaConf 加载失败。
        txt = txt.replace(
            "  supercategory: ${all_roboflow_supercategories.${string:${submitit.job_array.task_index}}}",
            "  supercategory: custom",
        )
        txt = txt.replace(
            "  num_images: 100 # Note: This is the number of images used for training. If null, all images are used.",
            "  num_images: null # Web job: null 表示使用全部训练图",
        )
        txt = txt.replace("  max_epochs: 20\n", f"  max_epochs: {int(max(1, epochs))}\n", 1)
        txt = txt.replace("  use_cluster: True\n", "  use_cluster: false\n", 1)
        txt = txt.replace("    num_tasks: 100\n", "    num_tasks: 0\n", 1)
        txt = txt.replace("  gpus_per_node: 2\n", "  gpus_per_node: 1\n", 1)
        if os.name == "nt":
            # Windows 下 nccl/forkserver 不可用，改为 gloo + spawn。
            txt = txt.replace("    backend: nccl\n", "    backend: gloo\n", 1)
            txt = txt.replace("  multiprocessing_context: forkserver\n", "  multiprocessing_context: spawn\n", 1)
        out_dir = self.project_root / "sam3" / "sam3" / "train" / "configs" / "web_jobs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{job.job_id}.yaml"
        out_path.write_text(txt, encoding="utf-8")
        return out_path

    def _find_yolo_dataset_root(self, extract_dir: Path) -> Path:
        """定位含 train/val/test 或平铺 images+labels 的数据根目录。"""
        if (extract_dir / "train" / "images").is_dir() and (
            extract_dir / "train" / "labels"
        ).is_dir():
            return extract_dir
        for child in sorted(extract_dir.iterdir()):
            if not child.is_dir():
                continue
            if (child / "train" / "images").is_dir() and (
                child / "train" / "labels"
            ).is_dir():
                return child
        return extract_dir

    def _pairs_from_split(self, root: Path, split: str) -> List[Tuple[Path, Path]]:
        """读取 YOLO 标准结构: {split}/images 与 {split}/labels。"""
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            return []
        pairs: List[Tuple[Path, Path]] = []
        for img in sorted(images_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            txt = labels_dir / f"{img.stem}.txt"
            if txt.is_file():
                pairs.append((img, txt))
        return pairs

    def _resolve_train_val_pairs(
        self, root: Path, train_ratio: float
    ) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]], str]:
        train_p = self._pairs_from_split(root, "train")
        val_p = self._pairs_from_split(root, "val")
        test_p = self._pairs_from_split(root, "test")

        if train_p and val_p:
            return train_p, val_p, "train/val 目录"
        if train_p and test_p and not val_p:
            return train_p, test_p, "train/test 目录（test 作验证集）"
        if train_p and not val_p and not test_p:
            if len(train_p) < 5:
                all_p = self._collect_pairs_flat(root)
                if len(all_p) >= 5:
                    random.shuffle(all_p)
                    idx = max(1, int(len(all_p) * train_ratio))
                    return all_p[:idx], all_p[idx:], "自动划分（仅 train 或平铺数据）"
            random.shuffle(train_p)
            idx = max(1, int(len(train_p) * train_ratio))
            return train_p[:idx], train_p[idx:], "自动划分（仅 train 目录）"

        all_pairs = self._collect_pairs_flat(root)
        if len(all_pairs) < 5:
            return [], [], "empty"
        random.shuffle(all_pairs)
        idx = max(1, int(len(all_pairs) * train_ratio))
        return all_pairs[:idx], all_pairs[idx:], "自动划分（兼容旧版平铺结构）"

    def _collect_pairs_flat(self, root: Path) -> List[Tuple[Path, Path]]:
        """兼容 zip 根目录直接放 images/labels 或混放的情况。"""
        pairs: List[Tuple[Path, Path]] = []
        for img in root.rglob("*"):
            if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
                continue
            txt = img.with_suffix(".txt")
            if txt.is_file():
                pairs.append((img, txt))
                continue
            parts = list(img.parts)
            if "images" in parts:
                idx = parts.index("images")
                parts[idx] = "labels"
                cand = Path(*parts).with_suffix(".txt")
                if cand.is_file():
                    pairs.append((img, cand))
        return pairs

    def _load_classes(self, root: Path) -> List[str]:
        cls_txt = list(root.rglob("classes.txt"))
        if cls_txt:
            return [x.strip() for x in cls_txt[0].read_text(encoding="utf-8").splitlines() if x.strip()]
        for yml in list(root.rglob("data.yaml")) + list(root.rglob("data.yml")):
            data = yaml.safe_load(yml.read_text(encoding="utf-8"))
            names = data.get("names")
            if isinstance(names, dict):
                return [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
            if isinstance(names, list):
                return [str(n) for n in names]
        return []

    def _build_split(
        self,
        pairs: List[Tuple[Path, Path]],
        out_dir: Path,
        classes: List[str],
        name_prefix: str = "",
    ) -> Dict:
        images = []
        annotations = []
        categories = [{"id": i, "name": n, "supercategory": "custom"} for i, n in enumerate(classes)]
        ann_id = 1
        img_id = 0
        for img_path, txt_path in pairs:
            safe_name = f"{name_prefix}{img_path.name}" if name_prefix else img_path.name
            target_img = out_dir / safe_name
            shutil.copy2(img_path, target_img)
            import cv2

            arr = cv2.imdecode(np.fromfile(str(target_img), dtype=np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                continue
            img_id += 1
            h, w = arr.shape[:2]
            images.append({"id": img_id, "file_name": target_img.name, "width": w, "height": h})
            for line in txt_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(float(parts[0]))
                if cid >= len(classes):
                    continue
                if len(parts) == 5:
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x = (cx - bw / 2) * w
                    y = (cy - bh / 2) * h
                    bw = bw * w
                    bh = bh * h
                elif len(parts) > 5 and len(parts) % 2 == 1:
                    pts = list(map(float, parts[1:]))
                    xs = [pts[i] * w for i in range(0, len(pts), 2)]
                    ys = [pts[i] * h for i in range(1, len(pts), 2)]
                    x, y = min(xs), min(ys)
                    bw, bh = max(xs) - x, max(ys) - y
                else:
                    continue
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cid,
                        "bbox": [x, y, bw, bh],
                        "area": float(max(0.0, bw * bh)),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
        return {"images": images, "annotations": annotations, "categories": categories}
