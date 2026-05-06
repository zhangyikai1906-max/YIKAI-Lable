const $ = (id) => document.getElementById(id);

/**
 * 使用 XMLHttpRequest 上传表单并报告进度（fetch 无法监听 upload 进度）。
 * @param {string} url
 * @param {FormData} formData
 * @param {(ratio: number) => void} [onProgress] 0~1，未知总长时不回调
 * @returns {Promise<any>}
 */
function postFormWithProgress(url, formData, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && typeof onProgress === "function") {
        onProgress(e.loaded / Math.max(e.total, 1));
      }
    };
    xhr.onload = () => {
      let data;
      try {
        data = JSON.parse(xhr.responseText || "{}");
      } catch (err) {
        reject(new Error(xhr.responseText || "响应解析失败"));
        return;
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(data);
      } else {
        reject(new Error(data.message || xhr.statusText || `HTTP ${xhr.status}`));
      }
    };
    xhr.onerror = () => reject(new Error("网络错误"));
    xhr.send(formData);
  });
}

function setInferProgress(visible, ratio) {
  const wrap = $("inferUploadProgress");
  const fill = $("inferProgressFill");
  const lab = $("inferProgressLabel");
  if (!wrap || !fill || !lab) return;
  if (visible) {
    wrap.hidden = false;
    const p = Math.round(Math.min(1, Math.max(0, ratio)) * 100);
    fill.style.width = `${p}%`;
    lab.textContent = `${p}%`;
  } else {
    wrap.hidden = true;
    fill.style.width = "0%";
    lab.textContent = "0%";
  }
}

function setTrainProgress(visible, ratio) {
  const wrap = $("trainUploadProgress");
  const fill = $("trainProgressFill");
  const lab = $("trainProgressLabel");
  if (!wrap || !fill || !lab) return;
  if (visible) {
    wrap.hidden = false;
    const p = Math.round(Math.min(1, Math.max(0, ratio)) * 100);
    fill.style.width = `${p}%`;
    lab.textContent = `${p}%`;
  } else {
    wrap.hidden = true;
    fill.style.width = "0%";
    lab.textContent = "0%";
  }
}

const cv = $("cv");
const ctx = cv.getContext("2d");
let img = null;
let modelLoaded = false;
let samEnabled = false;
let currentMode = "rect";
let currentTrainJobId = null;
let trainPollTimer = null;
const labelColors = new Map();
const palette = [
  "#67c23a", "#409eff", "#e6a23c", "#f56c6c", "#909399",
  "#9b59b6", "#1abc9c", "#e84393", "#2d98da", "#20bf6b",
];

const state = {
  annotations: [],
  drawing: false,
  start: null,
  temp: null,
  polyDraft: [],
};

async function refreshModelStatus() {
  const el = $("modelStatus");
  try {
    const r = await fetch("/api/model/status");
    const j = await r.json();
    if (j.loaded) {
      el.textContent = "模型已加载";
      el.className = "badge ok";
      modelLoaded = true;
    } else {
      el.textContent = "模型未就绪: " + (j.error || "请检查 SAM3_CHECKPOINT");
      el.className = "badge err";
      modelLoaded = false;
    }
  } catch (e) {
    el.textContent = "无法连接后端";
    el.className = "badge err";
    modelLoaded = false;
  }
  syncSamControls();
}

function drawImageFit() {
  if (!img) return;
  const maxW = Math.min(1000, window.innerWidth - 48);
  const scale = maxW / img.width;
  cv.width = Math.round(img.width * scale);
  cv.height = Math.round(img.height * scale);
  redrawCanvas();
}

function scaleX() {
  return cv.width / img.naturalWidth;
}

function scaleY() {
  return cv.height / img.naturalHeight;
}

function redrawCanvas() {
  if (!img) return;
  ctx.drawImage(img, 0, 0, cv.width, cv.height);
  const sx = scaleX();
  const sy = scaleY();
  ctx.lineWidth = 2;

  state.annotations.forEach((ann) => drawAnnotation(ann, sx, sy));
  if (state.temp) drawAnnotation(state.temp, sx, sy, true);

  if (state.polyDraft.length > 0) {
    ctx.strokeStyle = "#e6a23c";
    ctx.beginPath();
    const first = state.polyDraft[0];
    ctx.moveTo(first[0] * sx, first[1] * sy);
    for (let i = 1; i < state.polyDraft.length; i += 1) {
      ctx.lineTo(state.polyDraft[i][0] * sx, state.polyDraft[i][1] * sy);
    }
    ctx.stroke();
  }
}

function colorForLabel(label) {
  const key = (label || "object").trim() || "object";
  if (!labelColors.has(key)) {
    labelColors.set(key, palette[labelColors.size % palette.length]);
  }
  return labelColors.get(key);
}

function drawAnnotation(ann, sx, sy, isTemp = false) {
  const label = ann.label || "object";
  const color = isTemp ? "#f56c6c" : colorForLabel(label);
  ctx.strokeStyle = color;
  if (ann.type === "rectangle") {
    const [x, y, w, h] = ann.rect;
    ctx.strokeRect(x * sx, y * sy, w * sx, h * sy);
    drawLabelTag(label, x * sx, y * sy, color);
  } else if (ann.type === "point") {
    const [x, y] = ann.points[0];
    ctx.beginPath();
    ctx.arc(x * sx, y * sy, 4, 0, Math.PI * 2);
    ctx.stroke();
    drawLabelTag(label, x * sx + 6, y * sy - 6, color);
  } else if (ann.type === "polygon" || ann.type === "obb") {
    const pts = ann.points;
    if (!pts || pts.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(pts[0][0] * sx, pts[0][1] * sy);
    for (let i = 1; i < pts.length; i += 1) {
      ctx.lineTo(pts[i][0] * sx, pts[i][1] * sy);
    }
    ctx.closePath();
    ctx.stroke();
    drawLabelTag(label, pts[0][0] * sx, pts[0][1] * sy, color);
  }
}

function drawLabelTag(text, x, y, color) {
  const tag = text || "object";
  ctx.font = "12px Microsoft YaHei, sans-serif";
  const w = ctx.measureText(tag).width + 10;
  const h = 18;
  const tx = Math.max(2, x);
  const ty = Math.max(h + 2, y);
  ctx.fillStyle = color;
  ctx.fillRect(tx, ty - h, w, h);
  ctx.fillStyle = "#fff";
  ctx.fillText(tag, tx + 5, ty - 5);
}

function getImagePoint(evt) {
  const rect = cv.getBoundingClientRect();
  const x = (evt.clientX - rect.left) / scaleX();
  const y = (evt.clientY - rect.top) / scaleY();
  return [Math.max(0, x), Math.max(0, y)];
}

function makeRectFromPoints(p1, p2) {
  const x = Math.min(p1[0], p2[0]);
  const y = Math.min(p1[1], p2[1]);
  const w = Math.abs(p2[0] - p1[0]);
  const h = Math.abs(p2[1] - p1[1]);
  return [x, y, w, h];
}

function rectToObbPoints(rect) {
  const [x, y, w, h] = rect;
  return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]];
}

function syncSamControls() {
  $("promptInput").disabled = !samEnabled;
  $("btnInfer").disabled = !samEnabled || !img || !modelLoaded;
}

function currentLabel() {
  const v = $("labelInput").value.trim();
  return v || "object";
}

function setMode(mode) {
  currentMode = mode;
  state.polyDraft = [];
  document.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });
  $("hint").textContent = `当前模式: ${mode}`;
  redrawCanvas();
}

function exportAnnotations(format) {
  if (!img) return;
  const base = {
    imageWidth: img.naturalWidth,
    imageHeight: img.naturalHeight,
    imagePath: "uploaded_image",
  };
  if (format === "json") {
    const shapes = state.annotations.map((ann) => ({
      label: ann.label || "object",
      points: ann.points,
      shape_type: ann.type,
      flags: {},
      group_id: null,
    }));
    return JSON.stringify({ ...base, version: "web-0.1", flags: {}, shapes }, null, 2);
  }
  if (format === "yolo") {
    const w = img.naturalWidth;
    const h = img.naturalHeight;
    return state.annotations
      .map((ann) => {
        if (ann.type === "rectangle") {
          const [x, y, rw, rh] = ann.rect;
          const cx = (x + rw / 2) / w;
          const cy = (y + rh / 2) / h;
          return `0 ${cx.toFixed(6)} ${cy.toFixed(6)} ${(rw / w).toFixed(6)} ${(rh / h).toFixed(6)}`;
        }
        if (ann.type === "point") {
          const [x, y] = ann.points[0];
          return `0 ${(x / w).toFixed(6)} ${(y / h).toFixed(6)} 0.020000 0.020000`;
        }
        const flat = ann.points
          .map((p) => `${(p[0] / w).toFixed(6)} ${(p[1] / h).toFixed(6)}`)
          .join(" ");
        return `0 ${flat}`;
      })
      .join("\n");
  }
  const objects = state.annotations
    .map((ann) => {
      if (ann.type === "rectangle") {
        const [x, y, rw, rh] = ann.rect;
        return `<object><name>${ann.label || "object"}</name><bndbox><xmin>${Math.round(x)}</xmin><ymin>${Math.round(y)}</ymin><xmax>${Math.round(x + rw)}</xmax><ymax>${Math.round(y + rh)}</ymax></bndbox></object>`;
      }
      const poly = ann.points
        .map((p) => `<pt><x>${Math.round(p[0])}</x><y>${Math.round(p[1])}</y></pt>`)
        .join("");
      return `<object><name>${ann.label || "object"}</name><polygon>${poly}</polygon></object>`;
    })
    .join("");
  return `<?xml version="1.0" encoding="utf-8"?><annotation><size><width>${img.naturalWidth}</width><height>${img.naturalHeight}</height></size>${objects}</annotation>`;
}

function downloadText(content, filename) {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

$("fileInput").addEventListener("change", async (e) => {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  const fd = new FormData();
  fd.append("file", f);
  $("hint").textContent = "上传中…";
  setInferProgress(true, 0);
  let j;
  try {
    j = await postFormWithProgress("/api/infer/upload", fd, (r) => setInferProgress(true, r));
  } catch (err) {
    setInferProgress(false, 0);
    $("hint").textContent = err.message || "上传失败";
    $("jsonOut").textContent = JSON.stringify({ ok: false, message: String(err) }, null, 2);
    return;
  }
  setInferProgress(false, 0);
  $("jsonOut").textContent = JSON.stringify(j, null, 2);
  if (!j.ok) {
    $("hint").textContent = j.message || "上传失败";
    return;
  }
  img = new Image();
  img.onload = () => {
    state.annotations = [];
    state.polyDraft = [];
    state.temp = null;
    drawImageFit();
    $("hint").textContent = "图片已就绪，可输入提示词。";
    $("btnClear").disabled = false;
    $("btnExport").disabled = false;
    syncSamControls();
  };
  img.src = URL.createObjectURL(f);
});

$("btnInfer").addEventListener("click", async () => {
  if (!samEnabled) {
    alert("请先开启 SAM 智能辅助");
    return;
  }
  const prompt = $("promptInput").value.trim();
  if (!prompt) {
    alert("请输入提示词");
    return;
  }
  $("hint").textContent = "推理中…";
  const r = await fetch("/api/infer/text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  const j = await r.json();
  $("jsonOut").textContent = JSON.stringify(j, null, 2);
  if (j.ok && j.results) {
    const mapped = j.results.map((it) => {
      const label = currentLabel() || prompt;
      if (currentMode === "poly" && it.poly_pts && it.poly_pts.length >= 3) {
        return { type: "polygon", points: it.poly_pts, label };
      }
      if (currentMode === "rbox" && it.obb && it.obb.length >= 5) {
        const [cx, cy, w, h] = it.obb;
        return {
          type: "obb",
          points: rectToObbPoints([cx - w / 2, cy - h / 2, w, h]),
          label,
        };
      }
      return {
        type: "rectangle",
        rect: it.rect,
        points: rectToObbPoints(it.rect),
        label,
      };
    });
    state.annotations.push(...mapped);
    redrawCanvas();
    $("hint").textContent = `SAM 识别 ${mapped.length} 个目标`;
  } else {
    $("hint").textContent = j.message || "失败";
    redrawCanvas();
  }
});

$("btnClear").addEventListener("click", () => {
  state.annotations = [];
  state.polyDraft = [];
  state.temp = null;
  if (img) redrawCanvas();
});

$("btnExport").addEventListener("click", () => {
  const fmt = $("formatSelect").value;
  const content = exportAnnotations(fmt);
  const ext = fmt === "json" ? "json" : fmt === "xml" ? "xml" : "txt";
  downloadText(content, `annotation.${ext}`);
});

$("samSwitch").addEventListener("change", (e) => {
  samEnabled = e.target.checked;
  syncSamControls();
  $("hint").textContent = samEnabled ? "SAM 已开启，可输入提示词" : "SAM 已关闭，仅支持手动标注";
});

document.querySelectorAll(".mode-btn").forEach((btn) => {
  btn.addEventListener("click", () => setMode(btn.dataset.mode));
});

cv.addEventListener("mousedown", (evt) => {
  if (!img) return;
  const p = getImagePoint(evt);
  if (currentMode === "rect" || currentMode === "rbox") {
    state.drawing = true;
    state.start = p;
    state.temp = null;
  }
});

cv.addEventListener("mousemove", (evt) => {
  if (!img || !state.drawing || !state.start) return;
  const p = getImagePoint(evt);
  if (currentMode === "rect" || currentMode === "rbox") {
    const rect = makeRectFromPoints(state.start, p);
    state.temp = currentMode === "rect"
      ? { type: "rectangle", rect, points: rectToObbPoints(rect), label: "manual" }
      : { type: "obb", points: rectToObbPoints(rect), label: "manual" };
    redrawCanvas();
  }
});

cv.addEventListener("mouseup", (evt) => {
  if (!img) return;
  const p = getImagePoint(evt);
  if (currentMode === "rect" && state.start) {
    const rect = makeRectFromPoints(state.start, p);
    if (rect[2] > 2 && rect[3] > 2) {
      state.annotations.push({
        type: "rectangle",
        rect,
        points: rectToObbPoints(rect),
        label: currentLabel(),
      });
    }
  } else if (currentMode === "rbox" && state.start) {
    const rect = makeRectFromPoints(state.start, p);
    if (rect[2] > 2 && rect[3] > 2) {
      state.annotations.push({
        type: "obb",
        points: rectToObbPoints(rect),
        label: currentLabel(),
      });
    }
  }
  state.drawing = false;
  state.start = null;
  state.temp = null;
  redrawCanvas();
});

cv.addEventListener("click", (evt) => {
  if (!img) return;
  const p = getImagePoint(evt);
  if (currentMode === "point") {
    state.annotations.push({ type: "point", points: [p], label: currentLabel() });
    redrawCanvas();
  } else if (currentMode === "poly") {
    state.polyDraft.push(p);
    redrawCanvas();
  }
});

cv.addEventListener("dblclick", () => {
  if (currentMode !== "poly") return;
  if (state.polyDraft.length >= 3) {
    state.annotations.push({
      type: "polygon",
      points: [...state.polyDraft],
      label: currentLabel(),
    });
  }
  state.polyDraft = [];
  redrawCanvas();
});

cv.addEventListener("contextmenu", (evt) => {
  evt.preventDefault();
  if (currentMode === "poly") {
    state.polyDraft = [];
    redrawCanvas();
  }
});

window.addEventListener("resize", () => {
  if (img) drawImageFit();
});

refreshModelStatus();
syncSamControls();

async function pollTrainStatus() {
  if (!currentTrainJobId) return;
  const r = await fetch(`/api/train/status?job_id=${encodeURIComponent(currentTrainJobId)}`);
  const j = await r.json();
  if (!j.ok) return;
  $("trainHint").textContent = `任务 ${j.job_id}: ${j.status} - ${j.message}`;
  $("trainLog").textContent = (j.logs || []).join("\n") || "暂无日志";
  if (j.status === "done" || j.status === "failed") {
    if (trainPollTimer) {
      clearInterval(trainPollTimer);
      trainPollTimer = null;
    }
    if (j.output_ckpt) {
      $("trainHint").textContent += `；输出模型: ${j.output_ckpt}`;
    }
  }
}

$("btnTrainStart").addEventListener("click", async () => {
  const file = $("trainZipInput").files && $("trainZipInput").files[0];
  if (!file) {
    alert("请先选择 YOLO 压缩包");
    return;
  }
  const epochs = parseInt($("trainEpochs").value || "3", 10);
  const fd = new FormData();
  fd.append("file", file);
  $("trainHint").textContent = "正在上传并提交训练任务…";
  $("btnTrainStart").disabled = true;
  setTrainProgress(true, 0);
  let data;
  try {
    data = await postFormWithProgress(
      `/api/train/start?epochs=${epochs}`,
      fd,
      (r) => setTrainProgress(true, r)
    );
  } catch (err) {
    setTrainProgress(false, 0);
    $("btnTrainStart").disabled = false;
    $("trainHint").textContent = err.message || "上传或提交失败";
    $("jsonOut").textContent = JSON.stringify({ ok: false, message: String(err) }, null, 2);
    return;
  }
  setTrainProgress(false, 0);
  $("btnTrainStart").disabled = false;
  $("jsonOut").textContent = JSON.stringify(data, null, 2);
  if (!data.ok) {
    $("trainHint").textContent = data.message || "提交失败";
    return;
  }
  currentTrainJobId = data.job_id;
  $("trainHint").textContent = `任务已创建: ${currentTrainJobId}`;
  $("trainLog").textContent = "等待日志…";
  if (trainPollTimer) clearInterval(trainPollTimer);
  trainPollTimer = setInterval(pollTrainStatus, 2000);
  pollTrainStatus();
});
