# -*- coding: utf-8 -*-
"""
异步任务管理器：管理批量标注任务的生命周期。

状态机：
    pending → running → completed
                      ↘ failed
                      ↘ cancelled

特性：
  - 线程池执行（不阻塞事件循环）
  - WebSocket 实时广播进度
  - 任务持久化到磁盘（服务重启后可查询历史）
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from fastapi import WebSocket


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LabelTask:
    """单个标注任务。"""

    def __init__(self, job_id: str, job_dir: Path):
        self.job_id = job_id
        self.job_dir = job_dir
        self.status: TaskStatus = TaskStatus.PENDING
        self.progress: int = 0         # 已处理图数
        self.total: int = 0            # 总图数
        self.message: str = "等待中"
        self.error: Optional[str] = None
        self.created_at: float = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.config: Dict[str, Any] = {}
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or time.time()
        return end - self.started_at

    def eta_seconds(self) -> Optional[float]:
        if self.progress == 0 or self.total == 0:
            return None
        elapsed = self.elapsed_seconds()
        per_img = elapsed / self.progress
        remaining = self.total - self.progress
        return per_img * remaining

    def to_dict(self) -> dict:
        eta = self.eta_seconds()
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "progress": self.progress,
            "total": self.total,
            "percent": round(self.progress / self.total * 100, 1) if self.total > 0 else 0,
            "message": self.message,
            "error": self.error,
            "elapsed_s": round(self.elapsed_seconds(), 1),
            "eta_s": round(eta, 1) if eta is not None else None,
            "created_at": self.created_at,
            "config": self.config,
        }

    def save_meta(self) -> None:
        """持久化任务元数据到磁盘。"""
        meta_path = self.job_dir / "_task_meta.json"
        try:
            meta_path.write_text(
                json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass


class TaskManager:
    """
    全局任务管理器（FastAPI 生命周期内单例）。

    使用方式：
        manager = TaskManager(jobs_base_dir, max_workers=1)
        task = manager.create_task(job_id, job_dir, config)
        await manager.submit(task, run_fn)
    """

    def __init__(self, jobs_base_dir: Path, max_workers: int = 1):
        # max_workers=1：RTX 2060 显存有限，同时只跑一个 GPU 推理任务
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: Dict[str, LabelTask] = {}
        self._lock = threading.Lock()
        self._ws_clients: Dict[str, Set[WebSocket]] = {}  # job_id → WebSocket set
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.jobs_base_dir = jobs_base_dir

        self._load_existing_tasks()

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def create_task(
        self,
        job_id: str,
        job_dir: Path,
        config: Dict[str, Any],
    ) -> LabelTask:
        task = LabelTask(job_id, job_dir)
        task.config = config
        with self._lock:
            self._tasks[job_id] = task
        task.save_meta()
        return task

    def get_task(self, job_id: str) -> Optional[LabelTask]:
        return self._tasks.get(job_id)

    def list_tasks(self) -> List[Dict]:
        with self._lock:
            return [t.to_dict() for t in self._tasks.values()]

    def cancel_task(self, job_id: str) -> bool:
        task = self._tasks.get(job_id)
        if task and task.status == TaskStatus.RUNNING:
            task.request_cancel()
            task.status = TaskStatus.CANCELLED
            task.message = "已取消"
            task.finished_at = time.time()
            task.save_meta()
            self._broadcast(job_id, task.to_dict())
            return True
        return False

    async def submit(
        self,
        task: LabelTask,
        run_fn: Callable[[LabelTask], None],
    ) -> None:
        """
        提交任务到线程池异步执行。
        run_fn 接收 task 对象，通过 task.progress / task.message 更新进度。
        """
        loop = asyncio.get_running_loop()
        self._loop = loop

        def _wrapped():
            # 若任务被更新的提交覆盖，跳过执行
            if self._tasks.get(task.job_id) is not task:
                return

            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            task.message = "标注中..."
            task.save_meta()
            self._broadcast_sync(task.job_id, task.to_dict())

            try:
                run_fn(task)
                if not task.is_cancelled():
                    task.status = TaskStatus.COMPLETED
                    task.message = f"完成，共标注 {task.progress} 张图"
                    task.finished_at = time.time()
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = traceback.format_exc()
                task.message = f"失败: {e}"
                task.finished_at = time.time()
                import sys
                print(f"[TaskManager] 任务 {task.job_id} 执行失败:\n{task.error}", file=sys.stderr, flush=True)
            finally:
                task.save_meta()
                self._broadcast_sync(task.job_id, task.to_dict())

        loop.run_in_executor(self._executor, _wrapped)

    # ─────────────────────────────────────────
    # WebSocket 广播
    # ─────────────────────────────────────────

    async def ws_connect(self, job_id: str, ws: WebSocket) -> None:
        await ws.accept()
        with self._lock:
            if job_id not in self._ws_clients:
                self._ws_clients[job_id] = set()
            self._ws_clients[job_id].add(ws)

        # 立即发送当前状态
        task = self._tasks.get(job_id)
        if task:
            try:
                await ws.send_json(task.to_dict())
            except Exception:
                pass

    async def ws_disconnect(self, job_id: str, ws: WebSocket) -> None:
        with self._lock:
            if job_id in self._ws_clients:
                self._ws_clients[job_id].discard(ws)

    def _broadcast(self, job_id: str, data: dict) -> None:
        """在异步上下文中广播。"""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._async_broadcast(job_id, data), self._loop
            )

    def _broadcast_sync(self, job_id: str, data: dict) -> None:
        """在线程中广播（线程安全）。"""
        self._broadcast(job_id, data)

    async def _async_broadcast(self, job_id: str, data: dict) -> None:
        clients = self._ws_clients.get(job_id, set()).copy()
        dead: Set[WebSocket] = set()
        for ws in clients:
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        with self._lock:
            if job_id in self._ws_clients:
                self._ws_clients[job_id] -= dead

    # ─────────────────────────────────────────
    # 进度回调（供 LabelPipeline 调用）
    # ─────────────────────────────────────────

    def make_progress_callback(self, task: LabelTask) -> Callable:
        """
        返回一个进度回调函数，可直接传给 LabelPipeline.run()。
        """
        def callback(done: int, total: int, message: str = "") -> None:
            if task.is_cancelled():
                raise InterruptedError("任务已取消")
            task.progress = done
            task.total = total
            task.message = message
            task.save_meta()
            self._broadcast_sync(task.job_id, task.to_dict())

        return callback

    # ─────────────────────────────────────────
    # 历史任务加载
    # ─────────────────────────────────────────

    def _load_existing_tasks(self) -> None:
        label_jobs_dir = self.jobs_base_dir / "label_jobs"
        if not label_jobs_dir.exists():
            return
        for job_dir in label_jobs_dir.iterdir():
            if not job_dir.is_dir():
                continue
            meta_path = job_dir / "_task_meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                job_id = meta.get("job_id", job_dir.name)
                task = LabelTask(job_id, job_dir)
                task.status = TaskStatus(meta.get("status", "completed"))
                # 重启后：未完成的任务一律标记为已取消
                if task.status in (TaskStatus.RUNNING, TaskStatus.PENDING):
                    task.status = TaskStatus.CANCELLED
                    task.message = "服务重启，任务已取消"
                task.progress = meta.get("progress", 0)
                task.total = meta.get("total", 0)
                task.message = meta.get("message", "")
                task.config = meta.get("config", {})
                task.created_at = meta.get("created_at", 0)
                self._tasks[job_id] = task
            except Exception:
                pass
