# -*- coding: utf-8 -*-
"""
autoLable Web v2 服务入口（FastAPI）。

核心能力：
  - DINOv3 注意力图驱动的目标检测（无需 GroundingDINO）
  - SAM3 bbox prompt 精分割
  - DINOv3 特征聚类辅助预筛选
  - 批量图片上传（zip 包 / 多文件）
  - 多格式数据集导出（YOLO / COCO / VOC / LabelMe / Masks）
  - WebSocket 实时进度推送

启动（项目根目录）：
    set SAM3_CHECKPOINT=D:\\weights\\sam3.pt
    python -m uvicorn web.app_v2:app --host 0.0.0.0 --port 8081

浏览器：http://127.0.0.1:8081
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import List, Optional

from fastapi import (
    FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from web.models.dinov3_engine import DINOv3Engine
from web.models.sam_engine_v2 import SamEngineV2
from web.services.upload_service import UploadService
from web.services.label_pipeline import LabelPipeline, PipelineConfig
from web.services.export_service import ExportService, SUPPORTED_FORMATS
from web.services.task_manager import TaskManager, TaskStatus

# ─────────────────────────────────────────────
# 初始化
# ─────────────────────────────────────────────

_WEB_DIR = Path(__file__).resolve().parent
_STATIC_V2_DIR = _WEB_DIR / "static_v2"

app = FastAPI(
    title="autoLable Web v2",
    version="2.0.0",
    description="DINOv3 × SAM3 批量自动标注服务",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 模型引擎
_dinov3 = DINOv3Engine(variant="vits16")
_sam    = SamEngineV2()

# 服务
_upload_svc = UploadService(_WEB_DIR)
_task_mgr   = TaskManager(_WEB_DIR, max_workers=1)

# 模型加载进度追踪
_model_loading: dict = {
    "dinov3": {"state": "pending", "step": "", "elapsed": 0.0, "error": None},
    "sam3":   {"state": "pending", "step": "", "elapsed": 0.0, "error": None},
}


# ─────────────────────────────────────────────
# 启动
# ─────────────────────────────────────────────

def _load_dinov3_bg() -> None:
    """后台线程：加载 DINOv3 并更新进度。"""
    info = _model_loading["dinov3"]
    t0 = time.time()
    try:
        info["state"] = "loading"
        info["step"]  = f"扫描权重文件 (variant={_dinov3._variant})..."
        print(f"[模型加载] DINOv3 开始加载 variant={_dinov3._variant}", flush=True)
        result = _dinov3.load()
        info["elapsed"] = round(time.time() - t0, 1)
        if result.get("ok"):
            info["state"] = "ready"
            info["step"]  = f"已就绪 · {_dinov3._variant} · 特征维度 {_dinov3.feat_dim}"
            print(f"[模型加载] DINOv3 完成，耗时 {info['elapsed']}s", flush=True)
        else:
            info["state"] = "error"
            info["error"] = result.get("message", "未知错误")
            info["step"]  = "加载失败"
            print(f"[模型加载] DINOv3 失败: {info['error']}", flush=True)
    except Exception as e:
        info["state"]   = "error"
        info["error"]   = str(e)
        info["elapsed"] = round(time.time() - t0, 1)
        info["step"]    = "加载异常"
        print(f"[模型加载] DINOv3 异常: {e}", flush=True)


def _load_sam3_bg(sam_ckpt: str) -> None:
    """后台线程：加载 SAM3 并更新进度。"""
    info = _model_loading["sam3"]
    t0 = time.time()
    try:
        info["state"] = "loading"
        info["step"]  = "读取权重文件..."
        print(f"[模型加载] SAM3 开始加载 {sam_ckpt}", flush=True)
        result = _sam.load(sam_ckpt)
        info["elapsed"] = round(time.time() - t0, 1)
        if result.get("ok"):
            info["state"] = "ready"
            info["step"]  = f"已就绪 · 耗时 {info['elapsed']}s"
            print(f"[模型加载] SAM3 完成，耗时 {info['elapsed']}s", flush=True)
        else:
            info["state"] = "error"
            info["error"] = result.get("message", "未知错误")
            info["step"]  = "加载失败"
            print(f"[模型加载] SAM3 失败: {info['error']}", flush=True)
    except Exception as e:
        info["state"]   = "error"
        info["error"]   = str(e)
        info["elapsed"] = round(time.time() - t0, 1)
        info["step"]    = "加载异常"
        print(f"[模型加载] SAM3 异常: {e}", flush=True)


@app.on_event("startup")
async def startup() -> None:
    _task_mgr.set_event_loop(asyncio.get_running_loop())

    sam_ckpt = os.environ.get("SAM3_CHECKPOINT", r"D:\BaiduNetdiskDownload\权重\sam3.pt")

    # 检查 SAM3 权重是否存在
    if not os.path.isfile(sam_ckpt):
        _model_loading["sam3"]["state"] = "error"
        _model_loading["sam3"]["error"] = f"权重文件不存在: {sam_ckpt}"
        _model_loading["sam3"]["step"]  = "文件缺失，请设置 SAM3_CHECKPOINT 环境变量"

    # 后台并行加载两个模型，服务立即可用
    threading.Thread(target=_load_dinov3_bg, daemon=True).start()
    if os.path.isfile(sam_ckpt):
        threading.Thread(target=_load_sam3_bg, args=(sam_ckpt,), daemon=True).start()

    print("[startup] 服务已就绪，模型在后台加载中...", flush=True)


# ─────────────────────────────────────────────
# Pydantic 模型
# ─────────────────────────────────────────────

class JobConfigBody(BaseModel):
    class_names: List[str] = Field(default_factory=list, description="标注类别列表（用于导出标签）")
    # DINOv3 检测参数
    detect_mode: str = Field("attention", description="attention（注意力）/ patch_sim（少样本）")
    attn_threshold: float = Field(0.40, ge=0.05, le=0.95, description="注意力阈值，越高越严格")
    head_fusion: str = Field("mean", description="注意力头融合: mean / max / max_ent")
    min_area_ratio: float = Field(0.002, ge=0.0001, le=0.5)
    max_area_ratio: float = Field(0.95, ge=0.1, le=1.0)
    max_detections: int = Field(20, ge=1, le=100)
    # SAM3 分割参数
    mask_score_threshold: float = Field(0.5, ge=0.1, le=0.99)
    nms_iou_threshold: float = Field(0.5, ge=0.1, le=0.99)
    # 聚类
    use_dinov3_cluster: bool = False
    dinov3_n_clusters: int = Field(10, ge=2, le=100)
    # 导出
    export_formats: List[str] = Field(default_factory=lambda: ["yolo"])
    train_split_ratio: float = Field(0.8, ge=0.5, le=0.95)


class ModelLoadBody(BaseModel):
    sam3_checkpoint: Optional[str] = None
    dinov3_variant: Optional[str] = None   # "vits16" / "vits16plus" / "vitb16"


class CorrectAnnotationBody(BaseModel):
    image_name: str
    annotations: list


# ─────────────────────────────────────────────
# 健康检查 & 模型状态
# ─────────────────────────────────────────────

@app.get("/api/v2/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/api/v2/models/status")
def models_status():
    d = _model_loading["dinov3"]
    s = _model_loading["sam3"]
    return {
        "dinov3": {
            "loaded":   _dinov3.is_ready(),
            "state":    d["state"],          # pending | loading | ready | error
            "step":     d["step"],
            "elapsed":  d["elapsed"],
            "variant":  _dinov3._variant,
            "feat_dim": _dinov3.feat_dim,
            "error":    d["error"] or _dinov3.load_error,
        },
        "sam3": {
            "loaded":  _sam.is_ready(),
            "state":   s["state"],
            "step":    s["step"],
            "elapsed": s["elapsed"],
            "error":   s["error"] or _sam.load_error,
        },
        "all_ready": _dinov3.is_ready() and _sam.is_ready(),
    }


@app.post("/api/v2/models/load")
def models_load(body: ModelLoadBody):
    results = {}
    if body.sam3_checkpoint:
        results["sam3"] = _sam.load(body.sam3_checkpoint)
    if body.dinov3_variant:
        _dinov3._variant = body.dinov3_variant
        _dinov3.feat_dim = 768 if body.dinov3_variant == "vitb16" else 384
        results["dinov3"] = _dinov3.load()
    if not results:
        raise HTTPException(status_code=400, detail="未提供任何模型参数")
    return results


@app.get("/api/v2/models/dinov3/variants")
def dinov3_variants():
    """列出本地可用的 DINOv3 变体。"""
    from web.models.dinov3_engine import VARIANT_PATTERNS, _find_dinov3_dir
    available = []
    try:
        d = _find_dinov3_dir()
        for variant, prefix in VARIANT_PATTERNS.items():
            for f in d.iterdir():
                if f.name.startswith(prefix) and f.suffix == ".pth":
                    available.append({
                        "variant": variant,
                        "file": f.name,
                        "size_mb": round(f.stat().st_size / 1024 / 1024, 1),
                        "feat_dim": 768 if variant == "vitb16" else 384,
                    })
    except Exception:
        pass
    return {"ok": True, "variants": available}


# ─────────────────────────────────────────────
# 任务管理
# ─────────────────────────────────────────────

@app.post("/api/v2/jobs")
def create_job():
    job_id = _upload_svc.create_job()
    return {"ok": True, "job_id": job_id}


@app.get("/api/v2/jobs")
def list_jobs():
    return {"ok": True, "jobs": _task_mgr.list_tasks()}


@app.get("/api/v2/jobs/{job_id}")
def get_job(job_id: str):
    stats = _upload_svc.job_stats(job_id)
    task = _task_mgr.get_task(job_id)
    return {"ok": True, "stats": stats, "task": task.to_dict() if task else None}


@app.delete("/api/v2/jobs/{job_id}")
def delete_job(job_id: str):
    _upload_svc.delete_job(job_id)
    return {"ok": True}


@app.post("/api/v2/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if not _task_mgr.cancel_task(job_id):
        raise HTTPException(status_code=400, detail="任务不存在或未在运行中")
    return {"ok": True}


# ─────────────────────────────────────────────
# 图片上传
# ─────────────────────────────────────────────

@app.post("/api/v2/jobs/{job_id}/upload/zip")
async def upload_zip(job_id: str, file: UploadFile = File(...)):
    if _upload_svc.get_job_dir(job_id) is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="请上传 .zip 文件")
    data = await file.read()
    result = _upload_svc.handle_zip_upload(job_id, data)
    return {"ok": True, **result.to_dict()}


@app.post("/api/v2/jobs/{job_id}/upload/files")
async def upload_files(job_id: str, files: List[UploadFile] = File(...)):
    if _upload_svc.get_job_dir(job_id) is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    file_list = [(f.filename, await f.read()) for f in files]
    result = _upload_svc.handle_files_upload(job_id, file_list)
    return {"ok": True, **result.to_dict()}


# ─────────────────────────────────────────────
# 参考图上传（少样本模式）
# ─────────────────────────────────────────────

@app.post("/api/v2/jobs/{job_id}/references")
async def upload_references(job_id: str, files: List[UploadFile] = File(...)):
    """上传参考图（patch_sim 检测模式使用）。"""
    job_dir = _upload_svc.get_job_dir(job_id)
    if job_dir is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    ref_dir = job_dir / "references"
    ref_dir.mkdir(exist_ok=True)
    saved = []
    for f in files:
        data = await f.read()
        dest = ref_dir / f.filename
        dest.write_bytes(data)
        saved.append(f.filename)
    return {"ok": True, "saved": saved, "count": len(saved)}


# ─────────────────────────────────────────────
# 配置 & 启动标注
# ─────────────────────────────────────────────

@app.put("/api/v2/jobs/{job_id}/config")
def set_config(job_id: str, body: JobConfigBody):
    job_dir = _upload_svc.get_job_dir(job_id)
    if job_dir is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    (job_dir / "_config.json").write_text(
        json.dumps(body.dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"ok": True, "config": body.dict()}


@app.post("/api/v2/jobs/{job_id}/start")
async def start_labeling(job_id: str):
    """启动 DINOv3 × SAM3 批量标注（异步，WebSocket 跟踪进度）。"""
    job_dir = _upload_svc.get_job_dir(job_id)
    if job_dir is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    config_path = job_dir / "_config.json"
    if not config_path.exists():
        raise HTTPException(status_code=400, detail="请先 PUT /config 设置标注参数")

    body = JobConfigBody(**json.loads(config_path.read_text(encoding="utf-8")))

    if not _dinov3.is_ready():
        raise HTTPException(status_code=503, detail="DINOv3 未加载，请先 POST /api/v2/models/load")
    if not _sam.is_ready():
        raise HTTPException(status_code=503, detail="SAM3 未加载，请检查 SAM3_CHECKPOINT 环境变量")

    if not body.class_names:
        raise HTTPException(status_code=400, detail="请至少添加一个标注类别名称")

    image_paths = _upload_svc.list_images(job_id)
    if not image_paths:
        raise HTTPException(status_code=400, detail="未找到图片，请先上传")

    existing = _task_mgr.get_task(job_id)
    if existing and existing.status == TaskStatus.RUNNING:
        raise HTTPException(status_code=409, detail="任务已在运行中")

    # 加载参考图（少样本模式）
    reference_images = []
    if body.detect_mode == "patch_sim":
        ref_dir = job_dir / "references"
        if ref_dir.exists():
            from PIL import Image as PILImage
            from web.services.upload_service import SUPPORTED_IMAGE_EXTENSIONS
            for f in sorted(ref_dir.iterdir()):
                if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    try:
                        reference_images.append(PILImage.open(f).convert("RGB"))
                    except Exception:
                        pass

    pipeline_cfg = PipelineConfig(
        class_names=body.class_names,
        detect_mode=body.detect_mode,
        attn_threshold=body.attn_threshold,
        head_fusion=body.head_fusion,
        min_area_ratio=body.min_area_ratio,
        max_area_ratio=body.max_area_ratio,
        max_detections=body.max_detections,
        mask_score_threshold=body.mask_score_threshold,
        nms_iou_threshold=body.nms_iou_threshold,
        use_dinov3_cluster=body.use_dinov3_cluster,
        dinov3_n_clusters=body.dinov3_n_clusters,
        reference_images=reference_images,
    )

    task = _task_mgr.create_task(job_id, job_dir, body.dict())
    task.total = len(image_paths)
    pipeline = LabelPipeline(_dinov3, _sam)
    output_dir = job_dir / "results"
    progress_cb = _task_mgr.make_progress_callback(task)

    def run_fn(t):
        pipeline.run(
            image_paths=image_paths,
            output_dir=output_dir,
            config=pipeline_cfg,
            progress_callback=progress_cb,
        )
        # 标注完成后自动导出
        export_svc = ExportService(job_dir)
        for fmt in body.export_formats:
            try:
                export_svc.export(fmt, classes=body.class_names, split_ratio=body.train_split_ratio)
            except Exception:
                pass

    await _task_mgr.submit(task, run_fn)
    return {"ok": True, "job_id": job_id, "total_images": len(image_paths), "message": "标注任务已启动"}


# ─────────────────────────────────────────────
# 结果预览 & 修正
# ─────────────────────────────────────────────

@app.get("/api/v2/jobs/{job_id}/results")
def get_results(
    job_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    job_dir = _upload_svc.get_job_dir(job_id)
    if job_dir is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    results_dir = job_dir / "results"
    if not results_dir.exists():
        return {"ok": True, "total": 0, "results": []}
    all_files = [f for f in sorted(results_dir.glob("*.json")) if not f.name.startswith("_")]
    total = len(all_files)
    start = (page - 1) * page_size
    items = []
    for f in all_files[start: start + page_size]:
        try:
            items.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return {"ok": True, "total": total, "page": page, "page_size": page_size, "results": items}


@app.get("/api/v2/jobs/{job_id}/images/{image_name}")
def get_image(job_id: str, image_name: str):
    job_dir = _upload_svc.get_job_dir(job_id)
    if job_dir is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    img_path = job_dir / "images" / image_name
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="图片不存在")
    return FileResponse(str(img_path))


@app.get("/api/v2/jobs/{job_id}/annotated/{image_name}")
def get_annotated_image(job_id: str, image_name: str):
    """返回带检测框 + 多边形 + 标签的标注预览图（JPEG）。"""
    import cv2
    import numpy as np
    from fastapi.responses import Response

    job_dir = _upload_svc.get_job_dir(job_id)
    if job_dir is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    img_path = job_dir / "images" / image_name
    ann_path = job_dir / "results" / (Path(image_name).stem + ".json")

    if not img_path.exists():
        raise HTTPException(status_code=404, detail="图片不存在")

    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=500, detail="图片读取失败")

    if ann_path.exists():
        try:
            data = json.loads(ann_path.read_text(encoding="utf-8"))
            annotations = data.get("annotations", [])

            # 颜色列表（BGR）
            COLORS = [
                (0, 200, 255), (0, 255, 128), (255, 80, 80),
                (255, 200, 0), (180, 0, 255), (0, 180, 255),
                (255, 128, 0), (0, 255, 200),
            ]

            # ── Pass 1：先画所有半透明填充（避免后续轮廓被覆盖）
            overlay = img.copy()
            for i, ann in enumerate(annotations):
                color = COLORS[i % len(COLORS)]
                poly  = ann.get("poly_pts", [])
                if poly and len(poly) >= 3:
                    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [pts], color)
            img = cv2.addWeighted(overlay, 0.28, img, 0.72, 0)

            # ── Pass 2：画轮廓线 + 标签
            for i, ann in enumerate(annotations):
                color = COLORS[i % len(COLORS)]
                label = ann.get("label", "")
                score = ann.get("score", 0)
                poly  = ann.get("poly_pts", [])
                rect  = ann.get("rect_xywh", ann.get("box_xywh", []))

                if poly and len(poly) >= 3:
                    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                    # 多边形轮廓（双层：白色内描边 + 彩色外边）
                    cv2.polylines(img, [pts], isClosed=True, color=(255, 255, 255), thickness=3)
                    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

                    # 标签：定位在多边形最高点（y 最小处）
                    top_pt = pts.reshape(-1, 2)[pts.reshape(-1, 2)[:, 1].argmin()]
                    lx, ly = int(top_pt[0]), int(top_pt[1])
                elif rect and len(rect) == 4:
                    # 无多边形时退回到矩形框
                    rx, ry, rw, rh = [int(v) for v in rect]
                    cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), (255, 255, 255), 3)
                    cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), color, 2)
                    lx, ly = rx, ry
                else:
                    continue

                # 标签背景 + 文字
                text = f"{label}  {score:.2f}"
                font_scale = 0.45
                thickness  = 1
                (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                pad = 3
                tx = max(lx, 0)
                ty = max(ly - pad - 1, th + pad * 2)
                # 防止超出右边界
                if tx + tw + pad * 2 > img.shape[1]:
                    tx = img.shape[1] - tw - pad * 2
                cv2.rectangle(img, (tx, ty - th - pad), (tx + tw + pad * 2, ty + pad), color, -1)
                cv2.putText(img, text, (tx + pad, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        except Exception:
            pass

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/api/v2/jobs/{job_id}/heatmap/{image_name}")
def get_heatmap(
    job_id: str,
    image_name: str,
    head_fusion: str = Query("mean"),
):
    """返回 DINOv3 注意力热图（可视化调试用）。"""
    import io
    import cv2
    import numpy as np

    job_dir = _upload_svc.get_job_dir(job_id)
    if job_dir is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if not _dinov3.is_ready():
        raise HTTPException(status_code=503, detail="DINOv3 未加载")

    img_path = job_dir / "images" / image_name
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="图片不存在")

    from PIL import Image as PILImage
    from fastapi.responses import Response

    image = PILImage.open(img_path).convert("RGB")
    heat = _dinov3.detector.get_attention_heatmap(image, head_fusion=head_fusion)

    # 转为彩色热图
    heat_uint8 = (heat * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    img_np = np.array(image)[:, :, ::-1]  # RGB → BGR
    overlay = cv2.addWeighted(img_np, 0.5, heatmap_color, 0.5, 0)
    _, buf = cv2.imencode(".jpg", overlay)
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.post("/api/v2/jobs/{job_id}/correct")
def correct_annotation(job_id: str, body: CorrectAnnotationBody):
    job_dir = _upload_svc.get_job_dir(job_id)
    if job_dir is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    result_path = job_dir / "results" / (Path(body.image_name).stem + ".json")
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="标注结果不存在")
    data = json.loads(result_path.read_text(encoding="utf-8"))
    data["annotations"] = body.annotations
    data["manually_corrected"] = True
    result_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True}


# ─────────────────────────────────────────────
# 导出
# ─────────────────────────────────────────────

@app.post("/api/v2/jobs/{job_id}/export")
def export_dataset(
    job_id: str,
    fmt: str = Query("yolo", description=f"格式: {SUPPORTED_FORMATS}"),
    split_ratio: float = Query(0.8, ge=0.5, le=0.95),
):
    job_dir = _upload_svc.get_job_dir(job_id)
    if job_dir is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    config_path = job_dir / "_config.json"
    classes = None
    if config_path.exists():
        classes = json.loads(config_path.read_text(encoding="utf-8")).get("class_names")
    try:
        zip_path = ExportService(job_dir).export(fmt, classes=classes, split_ratio=split_ratio)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出失败: {e}")
    return {"ok": True, "format": fmt, "size_mb": round(zip_path.stat().st_size / 1024 / 1024, 2)}


@app.get("/api/v2/jobs/{job_id}/export/{fmt}/download")
def download_export(job_id: str, fmt: str):
    job_dir = _upload_svc.get_job_dir(job_id)
    if job_dir is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    zip_path = job_dir / "exports" / f"{fmt}.zip"
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail=f"尚未导出 {fmt} 格式")
    return FileResponse(str(zip_path), media_type="application/zip",
                        filename=f"{job_id}_{fmt}_dataset.zip")


@app.get("/api/v2/jobs/{job_id}/exports")
def list_exports(job_id: str):
    job_dir = _upload_svc.get_job_dir(job_id)
    if job_dir is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"ok": True, "exports": ExportService(job_dir).list_exports()}


# ─────────────────────────────────────────────
# WebSocket 进度
# ─────────────────────────────────────────────

@app.websocket("/api/v2/jobs/{job_id}/ws")
async def ws_progress(job_id: str, websocket: WebSocket):
    await _task_mgr.ws_connect(job_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await _task_mgr.ws_disconnect(job_id, websocket)


# ─────────────────────────────────────────────
# 静态前端
# ─────────────────────────────────────────────

@app.get("/")
def index():
    p = _STATIC_V2_DIR / "index.html"
    if not p.is_file():
        raise HTTPException(status_code=500, detail="前端未找到 web/static_v2/index.html")
    return FileResponse(str(p))


if _STATIC_V2_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_V2_DIR)), name="static_v2")
