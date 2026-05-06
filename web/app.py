# -*- coding: utf-8 -*-
"""
autoLable Web 服务入口：FastAPI + 静态前端。

启动（项目根目录）:
  set SAM3_CHECKPOINT=D:\\path\\to\\sam3.pt
  python -m uvicorn web.app:app --host 0.0.0.0 --port 8080

浏览器打开: http://127.0.0.1:8080
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from web.sam_engine import SamWebEngine
from web.train_service import TrainService

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="autoLable Web", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine = SamWebEngine()
_train_service = TrainService(Path(__file__).resolve().parents[1])


def default_checkpoint() -> str:
    return os.environ.get(
        "SAM3_CHECKPOINT",
        r"D:\BaiduNetdiskDownload\权重\sam3.pt",
    )


@app.on_event("startup")
def startup_load_model() -> None:
    ckpt = default_checkpoint()
    if os.path.isfile(ckpt):
        _engine.load_model(ckpt)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/model/status")
def model_status() -> dict:
    return {
        "loaded": _engine.is_ready(),
        "checkpoint": default_checkpoint(),
        "error": _engine.load_error,
    }


@app.post("/api/model/load")
def model_load(checkpoint_path: Optional[str] = Query(None)) -> dict:
    path = checkpoint_path or default_checkpoint()
    return _engine.load_model(path)


class TextBody(BaseModel):
    prompt: str


@app.post("/api/infer/upload")
async def infer_upload(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="空文件")
    return _engine.set_image(data)


@app.post("/api/infer/text")
def infer_text(body: TextBody) -> dict:
    return _engine.text_prompt(body.prompt)


@app.post("/api/train/start")
async def train_start(
    file: UploadFile = File(...),
    epochs: int = Query(3, ge=1, le=100),
    train_ratio: float = Query(0.8, ge=0.5, le=0.95),
) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="空文件")
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="请上传 YOLO 数据集 zip")
    job = _train_service.start_job_from_yolo_zip(data, epochs=epochs, train_ratio=train_ratio)
    return {"ok": True, "job_id": job.job_id, "status": job.status, "message": job.message}


@app.get("/api/train/status")
def train_status(job_id: str = Query(...)) -> dict:
    job = _train_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    return {
        "ok": True,
        "job_id": job.job_id,
        "status": job.status,
        "message": job.message,
        "output_ckpt": job.output_ckpt,
        "logs": job.log_lines[-200:],
    }


@app.get("/")
def index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=500, detail="前端未找到 web/static/index.html")
    return FileResponse(index_path)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
