<div align="center">

# 🏷️ autoLable

**基于 DINOv3 × SAM3 的 AI 智能图像自动标注系统**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![SAM3](https://img.shields.io/badge/AI-SAM3-FF6F00?logo=meta&logoColor=white)](https://github.com/facebookresearch/sam3)
[![DINOv3](https://img.shields.io/badge/AI-DINOv3-4A90D9?logo=meta&logoColor=white)](https://github.com/facebookresearch/dinov2)
[![License](https://img.shields.io/badge/License-Apache%202.0-green?logo=apache)](LICENSE)

无需人工标注，上传图片即可全自动生成训练数据集。  
DINOv3 无监督目标检测 + SAM3 精确分割，支持导出 YOLO / COCO / VOC 等主流格式。

[功能特性](#-功能特性) · [快速开始](#-快速开始) · [工作流程](#-工作流程) · [API 文档](#-api-文档) · [项目结构](#-项目结构) · [引用](#-引用)

</div>

---

## ✨ 功能特性

| 特性 | 说明 |
|------|------|
| 🔍 **无监督目标检测** | DINOv3 注意力图驱动，无需任何提示词，自动定位图中目标 |
| 🎯 **少样本匹配检测** | 上传参考图，通过 patch 特征相似度定位相同外观目标 |
| ✂️ **SAM3 精确分割** | 对检测 bbox 做多边形精分割，轮廓准确贴合目标边缘 |
| 📦 **批量处理** | 支持 zip 包或多文件批量上传，一键处理整个图片集 |
| 📤 **多格式导出** | YOLO / COCO / VOC / LabelMe / Masks 一键打包下载 |
| 🔄 **实时进度** | WebSocket 实时推送标注进度，前端动态展示处理状态 |
| ✏️ **人工修正** | AI 标注结果支持人工校正后回存，保证数据质量 |
| 🌐 **纯 Web 架构** | 浏览器即用，无需安装客户端，支持远程部署 |

---

## 🖥️ 界面预览

| 功能 | 效果图 |
|------|--------|
| 多边形标注 — 点选 | ![多边形点选](assets/img.png) |
| 多边形标注 — 点选结果 | ![多边形结果](assets/img_1.png) |
| OBB 旋转框 — 点选 | ![OBB点选](assets/img_2.png) |
| 目标检测 — 提示词结果 | ![检测提示词](assets/img_5.png) |

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- NVIDIA GPU（推荐，CPU 模式也可运行但速度较慢）
- CUDA 11.8+（GPU 模式）

### 第一步：克隆项目

```bash
git clone https://github.com/luohuabuxiema/autoLable.git
cd autoLable
```

### 第二步：安装 PyTorch

前往 [PyTorch 官网](https://pytorch.org/get-started/locally/) 根据 CUDA 版本选择安装命令，或使用阿里云镜像：

```bash
# CUDA 11.8（Windows 查看版本：nvidia-smi）
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 -f https://mirrors.aliyun.com/pytorch-wheels/cu118

# CUDA 12.1
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 -f https://mirrors.aliyun.com/pytorch-wheels/cu121
```

验证安装：

```python
import torch
print(torch.cuda.is_available())  # True 表示 GPU 可用
```

### 第三步：安装项目依赖

```bash
pip install -r requirements-web.txt
pip install -r requirements-v2.txt
```

### 第四步：安装 SAM3

```bash
cd sam3
pip install -e .
cd ..
```

### 第五步：下载模型权重

**SAM3 权重**（必须）：前往 [Hugging Face](https://huggingface.co/facebook/sam3/tree/main) 下载 `sam3.pt`。

**DINOv3 权重**（已内置）：项目 `dinov3/` 目录中已包含三种变体，无需额外下载。

### 第六步：启动服务

```bash
# Windows
set SAM3_CHECKPOINT=D:\your\path\to\sam3.pt
python -m uvicorn web.app_v2:app --host 0.0.0.0 --port 8081

# Linux / macOS
export SAM3_CHECKPOINT=/path/to/sam3.pt
python -m uvicorn web.app_v2:app --host 0.0.0.0 --port 8081
```

浏览器打开：`http://127.0.0.1:8081`

> **注意**：必须在**项目根目录**下运行启动命令。

---

## 🔄 工作流程

```
上传图片（zip 或多文件）
        ↓
设置标注参数（类别名、检测阈值等）
        ↓
启动自动标注
    ├── DINOv3 注意力图 → 目标 bbox 提议
    └── SAM3 bbox prompt → 精确多边形分割
        ↓
WebSocket 实时查看进度
        ↓
（可选）人工校正标注结果
        ↓
导出数据集（YOLO / COCO / VOC / LabelMe / Masks）
```

### 检测模式说明

**注意力图模式（`attention`，推荐）**

无需任何样本或提示词，DINOv3 通过分析 ViT 的自注意力权重，自动定位图像中的显著目标区域。适合大批量无标注图片的冷启动场景。

**少样本匹配模式（`patch_sim`）**

上传 1~5 张目标参考图，DINOv3 提取 patch 特征并计算相似度，定位图片中与参考图外观相似的目标。适合目标类别明确、外观一致的场景。

---

## 📖 使用指南

### 基本标注流程

1. **创建任务**：点击「新建任务」，获得唯一 `job_id`
2. **上传图片**：拖拽上传图片 zip 包或多张图片
3. **配置参数**：设置类别名称、检测模式和分割阈值
4. **启动标注**：点击「开始标注」，通过进度条实时追踪
5. **预览结果**：查看带标注框的预览图，确认标注质量
6. **人工校正**：对不准确的标注进行手动修正
7. **导出数据集**：选择格式导出，下载 zip 数据包

### 伪标签批处理（命令行）

对无标注图片批量生成 YOLO 弱标签数据集：

```bash
set SAM3_CHECKPOINT=D:\path\to\sam3.pt
python web/pseudo_label_batch.py \
  --input D:\raw_images \
  --output D:\pseudo_dataset \
  --prompt "car" \
  --class-name 车辆
```

输出结构：

```
pseudo_dataset/
├── train/images/  └── train/labels/
├── val/images/    └── val/labels/
├── classes.txt
└── data.yaml
```

---

## 📡 API 文档

服务启动后访问内置 Swagger 文档：`http://127.0.0.1:8081/docs`

### 核心接口速览

```
POST   /api/v2/jobs                          # 创建任务
POST   /api/v2/jobs/{job_id}/upload/zip      # 上传图片 zip
POST   /api/v2/jobs/{job_id}/upload/files    # 批量上传图片
POST   /api/v2/jobs/{job_id}/references      # 上传参考图（少样本模式）
PUT    /api/v2/jobs/{job_id}/config          # 设置标注参数
POST   /api/v2/jobs/{job_id}/start           # 启动标注
GET    /api/v2/jobs/{job_id}/results         # 获取标注结果（分页）
GET    /api/v2/jobs/{job_id}/annotated/{img} # 获取标注预览图
POST   /api/v2/jobs/{job_id}/correct         # 提交人工修正
POST   /api/v2/jobs/{job_id}/export?fmt=yolo # 导出数据集
GET    /api/v2/jobs/{job_id}/export/{fmt}/download  # 下载导出包
WS     /api/v2/jobs/{job_id}/ws              # WebSocket 进度推送
```

完整 API 文档见 [`web/README.md`](web/README.md)。

---

## 🗂️ 项目结构

```
autoLable/
├── web/                        # FastAPI Web 服务
│   ├── app.py                  # v1 入口（轻量，SAM3 + 文本提示词）
│   ├── app_v2.py               # v2 入口（完整，DINOv3 × SAM3）
│   ├── sam_engine.py           # v1 SAM3 推理引擎
│   ├── pseudo_label_batch.py   # 伪标签批处理脚本
│   ├── models/
│   │   ├── sam_engine_v2.py    # v2 SAM3 引擎
│   │   ├── dinov3_engine.py    # DINOv3 注意力检测引擎
│   │   └── grounding_dino_engine.py
│   ├── services/
│   │   ├── label_pipeline.py   # DINOv3 × SAM3 标注流水线
│   │   ├── export_service.py   # 多格式数据集导出
│   │   ├── upload_service.py   # 图片上传管理
│   │   └── task_manager.py     # 异步任务队列
│   └── static_v2/              # 前端静态资源
│
├── sam3/                       # SAM3 官方源码
├── dinov3/                     # DINOv3 本地权重文件
│   ├── dinov3_vits16_pretrain_*.pth       (~82 MB，默认)
│   ├── dinov3_vits16plus_pretrain_*.pth   (~110 MB)
│   └── dinov3_vitb16_pretrain_*.pth       (~327 MB)
│
├── scripts/
│   └── download_weights.py     # 权重下载工具
├── 01_json_to_unet.py          # JSON → U-Net Mask 转换
├── 02_split_yolo_dataset.py    # YOLO 数据集划分
├── 03_convert_and_split_yolo.py # 格式转换 + 划分
│
├── requirements-web.txt        # Web 基础依赖
├── requirements-v2.txt         # v2 完整依赖
├── LICENSE                     # Apache 2.0
└── SAM_LICENSE.txt             # SAM3 许可证
```

---

## ⚙️ 配置参数说明

标注任务的核心配置（`PUT /api/v2/jobs/{job_id}/config`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `class_names` | `[]` | 标注类别列表（至少填一个） |
| `detect_mode` | `"attention"` | `attention`（注意力图）/ `patch_sim`（少样本） |
| `attn_threshold` | `0.40` | 注意力阈值，越高越严格（0.05 ~ 0.95） |
| `head_fusion` | `"mean"` | 多头融合方式：`mean` / `max` / `max_ent` |
| `max_detections` | `20` | 每张图最多检测目标数 |
| `mask_score_threshold` | `0.5` | SAM3 分割置信度阈值 |
| `nms_iou_threshold` | `0.5` | NMS IoU 阈值 |
| `export_formats` | `["yolo"]` | 导出格式：`yolo`/`coco`/`voc`/`labelme`/`masks` |
| `train_split_ratio` | `0.8` | 训练集比例 |

---

## 🧠 DINOv3 模型变体

项目内置三种权重，启动时自动发现，按需切换：

| 变体 | 特征维度 | 大小 | 适用场景 |
|------|----------|------|----------|
| `vits16`（默认） | 384 | ~82 MB | RTX 2060+ 及以上，速度快 |
| `vits16plus` | 384 | ~110 MB | 特征表达更强（SwiGLU MLP） |
| `vitb16` | 768 | ~327 MB | 最强效果，需要较大显存 |

切换变体：`POST /api/v2/models/load {"dinov3_variant": "vitb16"}`

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

推荐扩展方向：

- **接入新模型**：`web/models/` 下新增引擎类，实现 `load()` / `detect()` 接口即可接入流水线
- **新增导出格式**：在 `web/services/export_service.py` 中扩展 `SUPPORTED_FORMATS`
- **前端功能增强**：`web/static_v2/` 基于原生 HTML/JS，可自由改造标注交互界面

---

## 📄 许可证

本项目采用 [Apache License 2.0](LICENSE) 开源协议。

本项目集成了 Meta AI 的 [SAM3](https://github.com/facebookresearch/sam3)，其许可证见 [SAM_LICENSE.txt](SAM_LICENSE.txt)。

---

## 📚 引用

如果本项目对你的研究或工作有帮助，欢迎 Star ⭐，并引用 SAM3 论文：

```bibtex
@misc{carion2025sam3segmentconcepts,
  title   = {SAM 3: Segment Anything with Concepts},
  author  = {Nicolas Carion and Laura Gustafson and Yuan-Ting Hu and others},
  year    = {2025},
  eprint  = {2511.16719},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url     = {https://arxiv.org/abs/2511.16719},
}
```

---

<div align="center">
Made with ❤️ · <a href="https://github.com/luohuabuxiema/autoLable">GitHub</a>
</div>
