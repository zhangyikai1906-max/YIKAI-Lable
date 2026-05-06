# -*- coding: utf-8 -*-
"""
DINOv3 引擎：特征提取 + 注意力驱动目标检测。

直接加载项目内 dinov3/ 目录下的三个本地权重，无需联网下载。

支持的变体：
  - vits16      dinov3_vits16_pretrain_*.pth     82.5MB  embed=384  标准MLP   ← 推荐
  - vits16plus  dinov3_vits16plus_pretrain_*.pth 109.6MB embed=384  SwiGLU MLP
  - vitb16      dinov3_vitb16_pretrain_*.pth     327MB   embed=768  标准MLP

架构特点：
  - RoPE 旋转位置编码（learnable periods）
  - storage_tokens：4 个全局上下文 token
  - patch_size=16，输入 224×224 → 14×14 patch grid
  - ViT-S/16+ 使用 SwiGLU MLP（w1/w2/w3）

两大功能：
  1. DINOv3Engine：批量提取图像特征 → KMeans 聚类 → 代表图筛选
  2. DINOv3Detector：注意力图驱动目标检测 → 输出 bbox 列表 → 供 SAM3 精分割

硬件建议（RTX 2060 6GB）：
  - 推荐 vits16：~82MB VRAM，与 SAM3 合计约 1.5GB，余量充足
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ─────────────────────────────────────────────────────────────
# 权重路径查找
# ─────────────────────────────────────────────────────────────

def _find_dinov3_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        d = parent / "dinov3"
        if d.is_dir():
            return d
    raise FileNotFoundError("未找到 dinov3/ 目录")


VARIANT_PATTERNS = {
    "vits16":     "dinov3_vits16_pretrain",
    "vits16plus": "dinov3_vits16plus_pretrain",
    "vitb16":     "dinov3_vitb16_pretrain",
}


def find_checkpoint(variant: str) -> Path:
    prefix = VARIANT_PATTERNS.get(variant)
    if not prefix:
        raise ValueError(f"未知变体: {variant}，可选: {list(VARIANT_PATTERNS)}")
    for f in _find_dinov3_dir().iterdir():
        if f.name.startswith(prefix) and f.suffix == ".pth":
            return f
    raise FileNotFoundError(f"未找到 {variant} 权重（前缀: {prefix}）")


# ─────────────────────────────────────────────────────────────
# DINOv3 ViT 架构
# ─────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int = 3, embed_dim: int = 384, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        B, C, Hp, Wp = x.shape
        return x.flatten(2).transpose(1, 2), Hp, Wp


class RoPEEmbed(nn.Module):
    def __init__(self, n_periods: int):
        super().__init__()
        self.periods = nn.Parameter(torch.ones(n_periods))

    def get_freqs_cis(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        freqs = 1.0 / self.periods.float().to(device)
        t_y = torch.arange(h, dtype=torch.float32, device=device)
        t_x = torch.arange(w, dtype=torch.float32, device=device)
        fy = torch.outer(t_y, freqs).unsqueeze(1).expand(-1, w, -1).reshape(h * w, -1)
        fx = torch.outer(t_x, freqs).unsqueeze(0).expand(h, -1, -1).reshape(h * w, -1)
        return torch.cat([
            torch.polar(torch.ones_like(fy), fy),
            torch.polar(torch.ones_like(fx), fx),
        ], dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor, n_patch: int):
    def rotate(x):
        xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        fc = freqs_cis.unsqueeze(0).unsqueeze(0)
        return torch.view_as_real(xc * fc).flatten(-2).type_as(x)
    q_out, k_out = q.clone(), k.clone()
    q_out[..., -n_patch:, :] = rotate(q[..., -n_patch:, :])
    k_out[..., -n_patch:, :] = rotate(k[..., -n_patch:, :])
    return q_out, k_out


class Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.register_buffer("qkv_bias_mask", torch.ones(embed_dim * 3), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        n_patch: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, C = x.shape
        eff_bias = self.qkv.bias * self.qkv_bias_mask if self.qkv.bias is not None else None
        qkv = F.linear(x, self.qkv.weight, eff_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if freqs_cis is not None and n_patch > 0:
            q, k = _apply_rope(q, k, freqs_cis, n_patch)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out, (attn if output_attentions else None)


class LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class StandardMLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        h = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, h)
        self.fc2 = nn.Linear(h, embed_dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class SwiGLUMLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        h = int(embed_dim * mlp_ratio)
        self.w1 = nn.Linear(embed_dim, h)
        self.w2 = nn.Linear(embed_dim, h)
        self.w3 = nn.Linear(h, embed_dim)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, use_swiglu: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads)
        self.ls1 = LayerScale(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = SwiGLUMLP(embed_dim) if use_swiglu else StandardMLP(embed_dim)
        self.ls2 = LayerScale(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        n_patch: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_w = self.attn(
            self.norm1(x), freqs_cis, output_attentions=output_attentions, n_patch=n_patch
        )
        x = x + self.ls1(attn_out)
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x, attn_w


class DINOv3ViT(nn.Module):
    """DINOv3 Vision Transformer，支持 vits16 / vits16plus / vitb16。"""

    VARIANT_CFG = {
        "vits16":     dict(embed_dim=384, num_heads=6,  num_blocks=12, use_swiglu=False),
        "vits16plus": dict(embed_dim=384, num_heads=6,  num_blocks=12, use_swiglu=True),
        "vitb16":     dict(embed_dim=768, num_heads=12, num_blocks=12, use_swiglu=False),
    }
    N_SPECIAL = 5  # 1 cls + 4 storage tokens

    def __init__(self, variant: str = "vits16"):
        super().__init__()
        cfg = self.VARIANT_CFG[variant]
        self.embed_dim = cfg["embed_dim"]
        self.num_heads = cfg["num_heads"]
        head_dim = self.embed_dim // self.num_heads
        n_periods = head_dim // 4

        self.patch_embed = PatchEmbed(embed_dim=self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.storage_tokens = nn.Parameter(torch.zeros(1, 4, self.embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, self.embed_dim))
        self.rope_embed = RoPEEmbed(n_periods)
        self.blocks = nn.ModuleList([
            Block(self.embed_dim, self.num_heads, cfg["use_swiglu"])
            for _ in range(cfg["num_blocks"])
        ])
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_all_tokens: bool = False,
        return_last_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[int, int]]]:
        """
        Returns:
            features:  (B, embed_dim) CLS token，或 (B, N, embed_dim) 全 tokens
            last_attn: (B, heads, N, N) 最后一层注意力权重（仅 return_last_attn=True 时）
            grid_hw:   (Hp, Wp) patch 网格尺寸（仅 return_last_attn=True 时）
        """
        B = x.shape[0]
        patches, Hp, Wp = self.patch_embed(x)
        tokens = torch.cat([
            self.cls_token.expand(B, -1, -1),
            self.storage_tokens.expand(B, -1, -1),
            patches,
        ], dim=1)

        n_patch = Hp * Wp
        freqs_cis = self.rope_embed.get_freqs_cis(Hp, Wp, x.device)

        last_attn = None
        for i, blk in enumerate(self.blocks):
            is_last = (i == len(self.blocks) - 1)
            tokens, attn_w = blk(
                tokens, freqs_cis,
                output_attentions=(is_last and return_last_attn),
                n_patch=n_patch,
            )
            if is_last:
                last_attn = attn_w

        tokens = self.norm(tokens)
        feats = tokens if return_all_tokens else tokens[:, 0]
        grid_hw = (Hp, Wp) if return_last_attn else None
        return feats, last_attn, grid_hw


def _load_dinov3(variant: str, checkpoint_path: Path) -> DINOv3ViT:
    model = DINOv3ViT(variant=variant)
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    for i, blk in enumerate(model.blocks):
        k = f"blocks.{i}.attn.qkv.bias_mask"
        if k in state:
            blk.attn.qkv_bias_mask.copy_(state[k].float())
    return model


# ─────────────────────────────────────────────────────────────
# DetectionResult（与 GroundingDINO 接口兼容）
# ─────────────────────────────────────────────────────────────

class DetectionResult:
    __slots__ = ("label", "score", "box_xyxy", "box_xywh")

    def __init__(self, label: str, score: float, box_xyxy: List[float]):
        self.label = label
        self.score = score
        self.box_xyxy = box_xyxy
        x1, y1, x2, y2 = box_xyxy
        self.box_xywh = [x1, y1, x2 - x1, y2 - y1]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "score": round(self.score, 4),
            "box_xyxy": [round(v, 2) for v in self.box_xyxy],
            "box_xywh": [round(v, 2) for v in self.box_xywh],
        }


# ─────────────────────────────────────────────────────────────
# DINOv3Detector：注意力图驱动的目标检测
# ─────────────────────────────────────────────────────────────

class DINOv3Detector:
    """
    基于 DINOv3 CLS 注意力图的目标检测器。

    原理：
      DINOv3（继承自 DINO 自监督训练范式）的 CLS token 对 patch tokens 的注意力权重
      天然形成"目标位置热图"——注意力高的区域就是图像中的显著目标。

      流程：
        1. DINOv3 最后一层 attention → CLS → patch 注意力 (heads, Hp, Wp)
        2. 多头融合（可选：平均 / 最大 / 最大熵头选择）
        3. 上采样到原图尺寸 → 归一化热图
        4. 自适应阈值（Otsu 或分位数）→ 二值图
        5. 连通域分析 → 滤除噪点 → 输出 bbox 列表

    与 SAM3 的配合：
      DINOv3 提供 bbox 提议 → SAM3 bbox prompt → 精确像素掩膜
      class_names 只用于标注（DINOv3 是视觉模型，不感知语义文本）

    支持两种检测模式（detect_mode 参数）：
      - "attention"  : 纯注意力图（无监督，检测所有显著目标）← 默认
      - "patch_sim"  : 参考图相似度（少样本，需提供 reference_images）
    """

    N_SPECIAL = 5  # cls + 4 storage

    def __init__(self, model: DINOv3ViT, device: str = "cuda"):
        self._model = model
        self._device = device
        self._lock = threading.Lock()

    # ── 注意力热图 ──────────────────────────────────────────

    def _get_attn_map(
        self,
        image: Image.Image,
        head_fusion: str = "mean",
    ) -> Tuple[np.ndarray, int, int]:
        """
        返回注意力热图（原图尺寸）和 patch grid 尺寸。

        head_fusion:
          "mean"    - 所有头平均
          "max"     - 所有头最大值
          "max_ent" - 选熵最大的头（最分散，通常最清晰）
        """
        W_orig, H_orig = image.size
        tensor = _preprocess([image]).to(self._device)

        with torch.inference_mode(), torch.autocast(device_type=self._device, dtype=torch.bfloat16):
            _, last_attn, (Hp, Wp) = self._model(
                tensor, return_last_attn=True
            )

        # last_attn: (1, heads, N, N)
        # CLS (index 0) → patch tokens (index N_SPECIAL : )
        attn = last_attn[0, :, 0, self.N_SPECIAL:].float()  # (heads, Hp*Wp)
        attn = attn.reshape(attn.shape[0], Hp, Wp)           # (heads, Hp, Wp)

        if head_fusion == "mean":
            heat = attn.mean(0)
        elif head_fusion == "max":
            heat = attn.max(0).values
        else:  # max_ent: head with highest entropy = most spread out attention
            ent = -(attn.reshape(attn.shape[0], -1) * (attn.reshape(attn.shape[0], -1) + 1e-8).log()).sum(-1)
            heat = attn[ent.argmax()]

        heat_np = heat.cpu().numpy().astype(np.float32)

        # 上采样到原图尺寸
        heat_up = cv2.resize(heat_np, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)

        # 归一化到 [0, 1]
        mn, mx = heat_up.min(), heat_up.max()
        if mx > mn:
            heat_up = (heat_up - mn) / (mx - mn)
        return heat_up, Hp, Wp

    # ── 主检测接口 ──────────────────────────────────────────

    def detect(
        self,
        image: Image.Image,
        class_names: Optional[List[str]] = None,
        attn_threshold: float = 0.4,
        min_area_ratio: float = 0.002,
        max_area_ratio: float = 0.95,
        max_detections: int = 20,
        head_fusion: str = "mean",
        detect_mode: str = "attention",
        reference_images: Optional[List[Image.Image]] = None,
    ) -> List[DetectionResult]:
        """
        对单张图片执行检测，返回 bbox 列表。

        Args:
            image:            输入图像
            class_names:      类别名列表（只用于标注，不影响检测逻辑）
            attn_threshold:   注意力阈值（0~1），越高越严格
            min_area_ratio:   最小目标面积（占图像比例），滤除噪点
            max_area_ratio:   最大目标面积，滤除背景
            max_detections:   最多返回多少个检测框
            head_fusion:      注意力头融合方式："mean" / "max" / "max_ent"
            detect_mode:      检测模式："attention"（默认）/ "patch_sim"（少样本）
            reference_images: detect_mode="patch_sim" 时提供参考图

        Returns:
            DetectionResult 列表，score 为该区域的平均注意力强度
        """
        if detect_mode == "patch_sim" and reference_images:
            return self._detect_by_similarity(
                image, reference_images, class_names,
                attn_threshold, min_area_ratio, max_area_ratio, max_detections,
            )
        return self._detect_by_attention(
            image, class_names, attn_threshold,
            min_area_ratio, max_area_ratio, max_detections, head_fusion,
        )

    def _detect_by_attention(
        self,
        image: Image.Image,
        class_names: Optional[List[str]],
        threshold: float,
        min_area_ratio: float,
        max_area_ratio: float,
        max_detections: int,
        head_fusion: str,
    ) -> List[DetectionResult]:
        W, H = image.size
        img_area = W * H

        heat, Hp, Wp = self._get_attn_map(image, head_fusion=head_fusion)

        # 自适应阈值：Otsu（若 threshold<=0）或固定值
        heat_uint8 = (heat * 255).astype(np.uint8)
        if threshold <= 0:
            thr_val, binary = cv2.threshold(heat_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(heat_uint8, int(threshold * 255), 255, cv2.THRESH_BINARY)

        # 形态学去噪：先腐蚀再膨胀
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # 连通域分析
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        results: List[DetectionResult] = []
        for comp_id in range(1, n_labels):  # 0 是背景
            x, y, bw, bh, area = stats[comp_id]
            area_ratio = area / img_area

            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue

            # 该连通域的平均注意力值作为 score
            mask = (labels == comp_id).astype(np.uint8)
            score = float(heat[mask > 0].mean())

            x1, y1 = float(x), float(y)
            x2, y2 = float(x + bw), float(y + bh)

            # 类别名：若只有一个类别直接用，多个类别无法区分则用 "object"
            label = class_names[0] if class_names and len(class_names) == 1 else "object"

            results.append(DetectionResult(label=label, score=score, box_xyxy=[x1, y1, x2, y2]))

        # 按 score 降序，取 top-N
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:max_detections]

    def _detect_by_similarity(
        self,
        image: Image.Image,
        references: List[Image.Image],
        class_names: Optional[List[str]],
        threshold: float,
        min_area_ratio: float,
        max_area_ratio: float,
        max_detections: int,
    ) -> List[DetectionResult]:
        """
        少样本检测：用参考图的 patch 特征在目标图中找相似区域。
        """
        W, H = image.size
        img_area = W * H

        # 提取参考图特征（取 patch tokens 平均作为 reference embedding）
        ref_tensors = _preprocess(references).to(self._device)
        with torch.inference_mode(), torch.autocast(device_type=self._device, dtype=torch.bfloat16):
            all_tokens, _, _ = self._model(ref_tensors, return_all_tokens=True)
        # patch tokens only (skip cls + storage)
        ref_patch_feats = all_tokens[:, self.N_SPECIAL:, :].float()       # (B_ref, Hp*Wp, D)
        ref_embed = ref_patch_feats.reshape(-1, ref_patch_feats.shape[-1]).mean(0)  # (D,)
        ref_embed = F.normalize(ref_embed.unsqueeze(0), dim=-1)           # (1, D)

        # 提取目标图 patch 特征
        tgt_tensor = _preprocess([image]).to(self._device)
        with torch.inference_mode(), torch.autocast(device_type=self._device, dtype=torch.bfloat16):
            tgt_tokens, _, (Hp, Wp) = self._model(tgt_tensor, return_all_tokens=True, return_last_attn=True)
        tgt_patch = tgt_tokens[0, self.N_SPECIAL:, :].float()             # (Hp*Wp, D)
        tgt_patch = F.normalize(tgt_patch, dim=-1)

        # 计算相似度热图
        sim = (tgt_patch @ ref_embed.T).squeeze(-1)                       # (Hp*Wp,)
        sim_map = sim.cpu().numpy().reshape(Hp, Wp)                        # (Hp, Wp)

        # 归一化 → 上采样
        mn, mx = sim_map.min(), sim_map.max()
        if mx > mn:
            sim_map = (sim_map - mn) / (mx - mn)
        heat = cv2.resize(sim_map.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

        # 二值化 + 连通域（复用 attention 逻辑）
        heat_uint8 = (heat * 255).astype(np.uint8)
        thr_val = max(1, int(threshold * 255))
        _, binary = cv2.threshold(heat_uint8, thr_val, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        results: List[DetectionResult] = []
        for comp_id in range(1, n_labels):
            x, y, bw, bh, area = stats[comp_id]
            if area / img_area < min_area_ratio or area / img_area > max_area_ratio:
                continue
            mask = (labels == comp_id).astype(np.uint8)
            score = float(heat[mask > 0].mean())
            label = class_names[0] if class_names and len(class_names) == 1 else "object"
            results.append(DetectionResult(
                label=label, score=score,
                box_xyxy=[float(x), float(y), float(x + bw), float(y + bh)],
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:max_detections]

    def get_attention_heatmap(
        self,
        image: Image.Image,
        head_fusion: str = "mean",
    ) -> np.ndarray:
        """返回归一化注意力热图 (H, W) float32，供前端可视化。"""
        heat, _, _ = self._get_attn_map(image, head_fusion=head_fusion)
        return heat

    def detect_batch(
        self,
        images: List[Image.Image],
        **kwargs,
    ) -> List[List[DetectionResult]]:
        """批量检测（逐张推理，复用模型）。"""
        return [self.detect(img, **kwargs) for img in images]


# ─────────────────────────────────────────────────────────────
# DINOv3Engine：特征提取 + 聚类
# ─────────────────────────────────────────────────────────────

_TRANSFORM_MEAN = [0.485, 0.456, 0.406]
_TRANSFORM_STD  = [0.229, 0.224, 0.225]


def _preprocess(images: List[Image.Image], size: int = 224) -> torch.Tensor:
    import torchvision.transforms.functional as TF
    tensors = []
    for img in images:
        img = img.convert("RGB")
        img = TF.resize(img, [size, size])
        t = TF.to_tensor(img)
        t = TF.normalize(t, _TRANSFORM_MEAN, _TRANSFORM_STD)
        tensors.append(t)
    return torch.stack(tensors)


class DINOv3Engine:
    """
    DINOv3 统一引擎：加载权重，提供特征提取、聚类、以及目标检测能力。

    使用方式：
        engine = DINOv3Engine(variant="vits16")
        engine.load()

        # 特征提取 + 聚类
        feats  = engine.extract([img1, img2, ...])
        labels = engine.cluster(feats, k=10)
        reps   = engine.representative_indices(feats, labels)

        # 目标检测（注意力模式）
        dets = engine.detector.detect(img, class_names=["car"])

        # 目标检测（少样本模式）
        dets = engine.detector.detect(
            img,
            class_names=["car"],
            detect_mode="patch_sim",
            reference_images=[ref1, ref2],
        )
    """

    SUPPORTED_VARIANTS = list(VARIANT_PATTERNS.keys())

    def __init__(self, variant: str = "vits16") -> None:
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(f"不支持的变体: {variant}，可选: {self.SUPPORTED_VARIANTS}")
        self._variant = variant
        self._model: Optional[DINOv3ViT] = None
        self._lock = threading.Lock()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_error: Optional[str] = None
        self.feat_dim: int = 768 if variant == "vitb16" else 384
        self.detector: Optional[DINOv3Detector] = None

    def is_ready(self) -> bool:
        return self._model is not None

    def load(self, checkpoint_path: Optional[str] = None) -> Dict:
        with self._lock:
            self.load_error = None
            try:
                ckpt = Path(checkpoint_path) if checkpoint_path else find_checkpoint(self._variant)
                if not ckpt.is_file():
                    raise FileNotFoundError(f"权重不存在: {ckpt}")
                model = _load_dinov3(self._variant, ckpt)
                model.to(self._device).eval()
                self._model = model
                self.detector = DINOv3Detector(model, device=self._device)
            except Exception as e:
                self.load_error = f"DINOv3 ({self._variant}) 加载失败: {e}"
                return {"ok": False, "message": self.load_error}

        return {
            "ok": True,
            "message": f"DINOv3 ({self._variant}) 加载成功（{self._device}），特征维度: {self.feat_dim}",
            "variant": self._variant,
            "feat_dim": self.feat_dim,
            "checkpoint": str(ckpt),
        }

    # ── 特征提取 ──────────────────────────────────────────

    def extract(
        self,
        images: List[Image.Image],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        if not self.is_ready():
            raise RuntimeError("DINOv3 未加载")
        all_feats: List[np.ndarray] = []
        with self._lock:
            for start in range(0, len(images), batch_size):
                tensors = _preprocess(images[start: start + batch_size]).to(self._device)
                with torch.inference_mode(), torch.autocast(device_type=self._device, dtype=torch.bfloat16):
                    feats, _, _ = self._model(tensors)
                feats = feats.float()
                if normalize:
                    feats = F.normalize(feats, dim=-1)
                all_feats.append(feats.cpu().numpy())
        return np.concatenate(all_feats, axis=0)

    # ── 聚类 ──────────────────────────────────────────────

    def cluster(self, features: np.ndarray, n_clusters: int = 10, random_state: int = 42) -> np.ndarray:
        try:
            from sklearn.cluster import KMeans, MiniBatchKMeans
        except ImportError:
            raise RuntimeError("请安装 scikit-learn: pip install scikit-learn")
        n_clusters = max(2, min(n_clusters, len(features)))
        km_cls = MiniBatchKMeans if len(features) > 5000 else KMeans
        return km_cls(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(features).astype(np.int32)

    def representative_indices(self, features: np.ndarray, cluster_labels: np.ndarray) -> List[int]:
        reps: List[int] = []
        for cid in np.unique(cluster_labels):
            mask = cluster_labels == cid
            idxs = np.where(mask)[0]
            sub = features[mask]
            best = int(np.argmin(np.sum((sub - sub.mean(0, keepdims=True)) ** 2, axis=1)))
            reps.append(int(idxs[best]))
        return sorted(reps)

    def top_k_similar(self, query: np.ndarray, gallery: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        q = query / (np.linalg.norm(query, axis=-1, keepdims=True) + 1e-8)
        g = gallery / (np.linalg.norm(gallery, axis=-1, keepdims=True) + 1e-8)
        sims = (q.reshape(1, -1) @ g.T).squeeze()
        top = np.argsort(sims)[::-1][:k]
        return [(int(i), float(sims[i])) for i in top]
