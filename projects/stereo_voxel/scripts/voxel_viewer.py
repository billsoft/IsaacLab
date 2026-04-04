"""语义体素 3D 可视化工具 v3
=========================
Python HTTP 后端 + Three.js 前端（本地离线，无需 CDN）。

功能：
  - 3D 体素三种可视化模式（键盘 1/2/3 切换）
      1. 语义模式：按语义类别着色
      2. Flow速度模式：按 XY 平面速度大小热力图着色（蓝→绿→红）
      3. 实例ID模式：每个实例唯一颜色
  - 同步左右眼立体图像
  - Meta 信息面板（相机位置、偏航角、NPC角速度、Flow统计）
  - 类别图例（语义模式，点击可切换显示/隐藏）
  - 速度色阶（Flow 模式）
  - 帧导航（滑块 + 键盘 ← →）

运行（仅需 numpy，不需要 Isaac Sim）：
    C:\\ProgramData\\anaconda3\\envs\\carla\\python.exe projects/stereo_voxel/scripts/voxel_viewer.py
    C:\\ProgramData\\anaconda3\\envs\\carla\\python.exe projects/stereo_voxel/scripts/voxel_viewer.py --data_dir X --port 8080
"""

import argparse
import http.server
import json
import os
import re
import socket
import sys
import threading
import webbrowser

import numpy as np

# ---------------------------------------------------------------------------
# semantic_classes 导入
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from semantic_classes import CLASS_COLORS, CLASS_NAMES, CLASS_NAMES_ZH, FREE, NUM_CLASSES, UNOBSERVED

# vendor 目录（three.min.js + OrbitControls.js）
VENDOR_DIR = os.path.join(SCRIPT_DIR, "vendor")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
DEFAULT_DATA = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "output_dng"))

parser = argparse.ArgumentParser(description="Voxel 3D Viewer")
parser.add_argument("--data_dir", default=DEFAULT_DATA, help="Dataset output directory")
parser.add_argument("--port", type=int, default=7860, help="HTTP port")
parser.add_argument("--no_browser", action="store_true")
args = parser.parse_args()

DATA_DIR = os.path.abspath(args.data_dir)


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def scan_frames(data_dir):
    left = os.path.join(data_dir, "left")
    if not os.path.isdir(left):
        return []
    ids = []
    for f in os.listdir(left):
        m = re.match(r"frame_(\d+)\.png$", f)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def load_voxels_json(data_dir, fid):
    """加载语义、实例、Flow 三种体素数据。

    Returns dict:
      voxels: [[x, y, z, sem_id, inst_id], ...]   所有占用体素（排除 FREE/UNOBSERVED）
      flow:   [[x, y, z, vx, vy, speed], ...]     flow_mask=1 的动态体素
      stats:  {str(class_id): count, "255": unobs, "total": total}
    """
    sem_path = os.path.join(data_dir, "voxel", f"frame_{fid:06d}_semantic.npz")
    if not os.path.isfile(sem_path):
        return {"voxels": [], "flow": [], "stats": {}}

    sem = np.load(sem_path)["data"]  # (NX, NY, NZ) uint8

    # 实例 ID
    inst_path = os.path.join(data_dir, "voxel", f"frame_{fid:06d}_instance.npz")
    if os.path.isfile(inst_path):
        inst = np.load(inst_path)["data"].astype(np.int32)
    else:
        inst = np.zeros_like(sem, dtype=np.int32)

    # Flow 数据：(NX, NY, NZ, 2) float16 [vx, vy], mask uint8
    flow_path = os.path.join(data_dir, "voxel", f"frame_{fid:06d}_flow.npz")
    flow_list = []
    if os.path.isfile(flow_path):
        fd = np.load(flow_path)
        flow_arr = fd["flow"].astype(np.float32)   # (NX,NY,NZ,2)
        flow_mask = fd["flow_mask"]                # (NX,NY,NZ) uint8
        flow_idx = np.argwhere(flow_mask == 1)
        for r in flow_idx:
            i, j, k = int(r[0]), int(r[1]), int(r[2])
            vx = float(flow_arr[i, j, k, 0])
            vy = float(flow_arr[i, j, k, 1])
            speed = float(np.sqrt(vx ** 2 + vy ** 2))
            flow_list.append([i, j, k, round(vx, 4), round(vy, 4), round(speed, 4)])

    # 占用体素（排除 FREE 和 UNOBSERVED）
    mask = (sem != FREE) & (sem != UNOBSERVED)
    idx = np.argwhere(mask)
    voxels = []
    for r in idx:
        i, j, k = int(r[0]), int(r[1]), int(r[2])
        voxels.append([i, j, k, int(sem[i, j, k]), int(inst[i, j, k])])

    # 统计
    stats = {}
    for cid in range(NUM_CLASSES):
        c = int(np.sum(sem == cid))
        if c > 0:
            stats[str(cid)] = c
    stats["255"] = int(np.sum(sem == UNOBSERVED))
    stats["total"] = int(sem.size)

    return {"voxels": voxels, "flow": flow_list, "stats": stats}


def load_meta(data_dir, fid):
    p = os.path.join(data_dir, "meta", f"frame_{fid:06d}.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def class_info_js():
    d = {}
    for cid in range(NUM_CLASSES):
        r, g, b = CLASS_COLORS[cid]
        d[cid] = {"name": CLASS_NAMES[cid], "zh": CLASS_NAMES_ZH[cid], "rgb": [r, g, b]}
    return json.dumps(d)


# ---------------------------------------------------------------------------
# 自动找可用端口
# ---------------------------------------------------------------------------
def find_free_port(start, tries=20):
    for p in range(start, start + tries):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", p))
            s.close()
            return p
        except OSError:
            continue
    return None


# ---------------------------------------------------------------------------
# HTML 前端
# ---------------------------------------------------------------------------
FRONTEND = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<title>Voxel Viewer</title>
<style>
:root{--bg:#111827;--panel:#1f2937;--border:#374151;--accent:#f59e0b;--text:#e5e7eb;--dim:#9ca3af}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--text);
     overflow:hidden;height:100vh;display:flex;flex-direction:column}

/* 顶部工具栏 */
.bar{display:flex;align-items:center;gap:8px;padding:6px 14px;background:var(--panel);
     border-bottom:1px solid var(--border);height:44px;flex-shrink:0;flex-wrap:wrap}
.bar label{font-size:12px;color:var(--dim)}
.bar input[type=text]{background:#111827;border:1px solid var(--border);color:var(--text);
     padding:3px 8px;border-radius:4px;width:290px;font-size:12px}
.bar button{background:#374151;border:1px solid var(--border);color:var(--text);
     padding:4px 10px;border-radius:4px;cursor:pointer;font-size:12px;transition:background .15s}
.bar button:hover{background:#4b5563}
.bar .sep{width:1px;height:22px;background:var(--border)}
#slider{width:170px;accent-color:var(--accent)}
#flab{font-weight:700;color:var(--accent);min-width:130px;font-size:12px}

/* 模式切换按钮 */
.mode-btn{background:#1f2937 !important;border-color:#4b5563 !important}
.mode-btn.active{background:#f59e0b !important;color:#111 !important;font-weight:700;border-color:#f59e0b !important}

/* 主布局 */
.main{display:flex;flex:1;overflow:hidden}

/* 左侧面板 */
.side{width:300px;min-width:230px;background:var(--panel);border-right:1px solid var(--border);
      overflow-y:auto;padding:8px;flex-shrink:0;font-size:11px}
.side h3{font-size:10px;color:var(--dim);margin:10px 0 3px;text-transform:uppercase;
         letter-spacing:1px;border-bottom:1px solid var(--border);padding-bottom:3px}
.side h3:first-child{margin-top:2px}
.side img{width:100%;border-radius:4px;border:1px solid var(--border);display:block;
          background:#0a0f1a;min-height:40px}

/* 统计框 */
.infobox,.stats{margin-top:4px;padding:6px;background:#0d1117;border-radius:4px;border:1px solid #1e2732}
.srow,.irow{display:flex;align-items:center;gap:5px;padding:2px 0;line-height:1.4}
.sw{width:9px;height:9px;border-radius:2px;flex-shrink:0}
.sn,.ik{color:var(--dim);min-width:78px;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.sc,.iv{font-family:monospace;color:var(--text);font-size:11px}
.sc{color:var(--accent)}
.divider{border-top:1px solid var(--border);margin-top:3px;padding-top:3px}

/* 3D视口 */
.view{flex:1;position:relative;overflow:hidden}
canvas{display:block;width:100%;height:100%}

/* 图例 */
.legend{position:absolute;bottom:10px;left:10px;background:rgba(17,24,39,.95);
        border:1px solid var(--border);border-radius:6px;padding:8px 12px;
        font-size:11px;max-height:55vh;overflow-y:auto;user-select:none;min-width:150px}
.legend h4{color:var(--accent);margin-bottom:5px;font-size:12px}
.lg{display:flex;align-items:center;gap:5px;padding:2px 0;cursor:pointer}
.lg:hover{opacity:.75}
.lg.off .ls{opacity:.25}
.lg.off .lt{text-decoration:line-through;opacity:.35}
.ls{width:11px;height:11px;border-radius:2px;border:1px solid rgba(255,255,255,.12);flex-shrink:0}
.lt{white-space:nowrap;color:var(--text)}

/* 速度色阶条 */
.grad-wrap{margin-top:4px}
.grad-strip{width:130px;height:10px;border-radius:2px;
            background:linear-gradient(to right,hsl(240,100%,50%),hsl(120,100%,50%),hsl(0,100%,50%))}
.grad-labs{display:flex;justify-content:space-between;width:130px;
           font-size:10px;color:var(--dim);margin-top:2px}

/* HUD */
.hud{position:absolute;top:10px;left:10px;background:rgba(17,24,39,.85);border-radius:4px;
     padding:5px 10px;font-size:11px;color:var(--dim);pointer-events:none;border:1px solid var(--border)}
#ld{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
    font-size:15px;color:var(--accent);background:rgba(17,24,39,.9);padding:8px 16px;border-radius:6px}

/* 快捷键提示 */
.keys{position:absolute;bottom:10px;right:10px;background:rgba(17,24,39,.75);
      border-radius:4px;padding:5px 8px;font-size:10px;color:var(--dim);pointer-events:none}
</style></head>
<body>

<div class="bar">
  <label>目录:</label>
  <input type="text" id="dirIn" value="">
  <button onclick="chDir()">加载</button>
  <div class="sep"></div>
  <button onclick="go(-1)" title="上一帧 [←]">&#9664;</button>
  <input type="range" id="slider" min="0" max="0" value="0" oninput="go(0,+this.value)">
  <button onclick="go(+1)" title="下一帧 [→]">&#9654;</button>
  <span id="flab">&mdash;</span>
  <div class="sep"></div>
  <button class="mode-btn active" id="btn-sem" onclick="setMode('sem')" title="键盘 1">语义</button>
  <button class="mode-btn" id="btn-flow" onclick="setMode('flow')" title="键盘 2">Flow速度</button>
  <button class="mode-btn" id="btn-inst" onclick="setMode('inst')" title="键盘 3">实例ID</button>
</div>

<div class="main">
  <div class="side">
    <h3>左眼图像</h3>
    <img id="imgL" alt="left">
    <h3>右眼图像</h3>
    <img id="imgR" alt="right">
    <h3 id="statsTitle">统计 — 语义</h3>
    <div class="stats" id="statsBox"><div style="color:var(--dim)">加载中...</div></div>
    <h3>Meta 信息</h3>
    <div class="infobox" id="metaBox"><div style="color:var(--dim)">加载中...</div></div>
  </div>

  <div class="view" id="vp">
    <canvas id="c"></canvas>
    <div class="hud" id="hud">—</div>
    <div class="legend" id="legend"></div>
    <div id="ld">Loading&hellip;</div>
    <div class="keys">← → 切帧 &nbsp;|&nbsp; 1/2/3 切模式 &nbsp;|&nbsp; 拖拽旋转 滚轮缩放 右键平移</div>
  </div>
</div>

<script src="/vendor/three.min.js"></script>
<script src="/vendor/OrbitControls.js"></script>
<script>
/* =========================================================
   常量 & 全局状态
   ========================================================= */
const CI = __CLASS_JSON__;
const VS = 0.1, NX = 72, NY = 60, NZ = 32, CX = 36, CY = 30, ZG = 7;

let frames = [], ci = 0, hidden = new Set();
let curVox = [], curFlow = [], curMode = 'sem';
let mesh = null;
window._lastStats = {};

/* =========================================================
   Three.js 初始化
   ========================================================= */
const cvs = document.getElementById('c');
const ren = new THREE.WebGLRenderer({ canvas: cvs, antialias: true });
ren.setPixelRatio(Math.min(devicePixelRatio, 2));
ren.setClearColor(0x0d1117);

const sc = new THREE.Scene();
const cam = new THREE.PerspectiveCamera(45, 1, 0.05, 300);
cam.position.set(6, 5, 6);

const ctrl = new THREE.OrbitControls(cam, cvs);
ctrl.enableDamping = true;
ctrl.dampingFactor = 0.08;
ctrl.target.set(0, 0.8, 0);

/* 场景辅助元素 — 仅用环境光（不影响 MeshBasicMaterial） */
sc.add(new THREE.AmbientLight(0xffffff, 1.0));
sc.add(new THREE.GridHelper(8, 40, 0x222e3f, 0x161f2b));

/* 体素网格包围框线框 */
(function () {
  const g = new THREE.BoxGeometry(NX * VS, NZ * VS, NY * VS);
  const e = new THREE.EdgesGeometry(g);
  const l = new THREE.LineSegments(e, new THREE.LineBasicMaterial({ color: 0x2d3f55, transparent: true, opacity: 0.6 }));
  l.position.set(0, -ZG * VS + NZ * VS / 2, 0);
  sc.add(l);
})();

/* 坐标轴（体素网格角落） */
const ax = new THREE.AxesHelper(1.0);
ax.position.set(-NX * VS / 2, -ZG * VS, -NY * VS / 2);
sc.add(ax);

/* 地面层标记线 */
(function () {
  const y = (-ZG + 0.5) * VS;  // z=7 体素中心高度
  const pts = [
    new THREE.Vector3(-NX * VS / 2, y, -NY * VS / 2),
    new THREE.Vector3(NX * VS / 2, y, -NY * VS / 2),
    new THREE.Vector3(NX * VS / 2, y, NY * VS / 2),
    new THREE.Vector3(-NX * VS / 2, y, NY * VS / 2),
    new THREE.Vector3(-NX * VS / 2, y, -NY * VS / 2),
  ];
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  sc.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0x3d5270, transparent: true, opacity: 0.5 })));
})();

/* =========================================================
   体素几何（使用 MeshBasicMaterial，颜色准确无光照偏差）
   ========================================================= */
const boxG = new THREE.BoxGeometry(VS * 0.87, VS * 0.87, VS * 0.87);

/* 体素索引 → Three.js 世界坐标
   体素坐标系：i=X轴, j=Y轴(深度), k=Z轴(高度)
   Three.js：X=右, Y=上, Z=前（右手系）
*/
function v2p(i, j, k) {
  return new THREE.Vector3(
    (i - CX + 0.5) * VS,   // X
    (k - ZG + 0.5) * VS,   // Y（高度，ZG 为地面层）
    (j - CY + 0.5) * VS    // Z（深度）
  );
}

/* =========================================================
   颜色函数
   ========================================================= */

/* 语义颜色：直接用类别 RGB（MeshBasicMaterial 不受光照影响） */
function semClr(cid) {
  const rgb = CI[cid]?.rgb || [128, 128, 128];
  return new THREE.Color(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255);
}

/* Flow速度颜色：蓝(0)→绿(mid)→红(max)，HSL hue 240°→0° */
function flowClr(speed, maxSpd) {
  const t = (maxSpd > 0.001) ? Math.min(speed / maxSpd, 1.0) : 0.0;
  const c = new THREE.Color();
  c.setHSL((1.0 - t) * 0.667, 1.0, 0.5);  // 240°→0°
  return c;
}

/* 实例ID颜色：黄金角哈希 → 饱和 HSL */
function instClr(id) {
  const c = new THREE.Color();
  if (!id || id === 0) {
    c.setRGB(0.18, 0.22, 0.28);  // 静态/背景：暗灰
    return c;
  }
  const h = ((id * 2654435761) >>> 0) % 360;
  c.setHSL(h / 360, 0.82, 0.55);
  return c;
}

/* =========================================================
   体素渲染
   ========================================================= */
function rebuild() {
  /* 清除旧网格 */
  if (mesh) {
    sc.remove(mesh);
    mesh.geometry.dispose();
    mesh.material.dispose();
    mesh = null;
  }

  const dm = new THREE.Object3D();
  const mat = new THREE.MeshBasicMaterial();

  if (curMode === 'sem') {
    const vis = curVox.filter(v => !hidden.has(v[3]));
    updHud(vis.length, curVox.length);
    if (!vis.length) { mkLeg(); return; }

    const im = new THREE.InstancedMesh(boxG, mat, vis.length);
    for (let n = 0; n < vis.length; n++) {
      const [i, j, k, cid] = vis[n];
      dm.position.copy(v2p(i, j, k)); dm.updateMatrix();
      im.setMatrixAt(n, dm.matrix);
      im.setColorAt(n, semClr(cid));
    }
    im.instanceMatrix.needsUpdate = true;
    if (im.instanceColor) im.instanceColor.needsUpdate = true;
    sc.add(im); mesh = im;

  } else if (curMode === 'flow') {
    const pts = curFlow;
    updHud(pts.length, curVox.length);
    if (!pts.length) { mkLeg(); return; }

    const maxSpd = Math.max(...pts.map(f => f[5]), 0.001);
    const im = new THREE.InstancedMesh(boxG, mat, pts.length);
    for (let n = 0; n < pts.length; n++) {
      const [i, j, k, , , spd] = pts[n];
      dm.position.copy(v2p(i, j, k)); dm.updateMatrix();
      im.setMatrixAt(n, dm.matrix);
      im.setColorAt(n, flowClr(spd, maxSpd));
    }
    im.instanceMatrix.needsUpdate = true;
    if (im.instanceColor) im.instanceColor.needsUpdate = true;
    sc.add(im); mesh = im;

  } else {  // inst
    const pts = curVox;
    updHud(pts.length, pts.length);
    if (!pts.length) { mkLeg(); return; }

    const im = new THREE.InstancedMesh(boxG, mat, pts.length);
    for (let n = 0; n < pts.length; n++) {
      const [i, j, k, , iid] = pts[n];
      dm.position.copy(v2p(i, j, k)); dm.updateMatrix();
      im.setMatrixAt(n, dm.matrix);
      im.setColorAt(n, instClr(iid));
    }
    im.instanceMatrix.needsUpdate = true;
    if (im.instanceColor) im.instanceColor.needsUpdate = true;
    sc.add(im); mesh = im;
  }

  mkLeg();
}

function updHud(vis, total) {
  const modeStr = { sem: '语义', flow: 'Flow速度', inst: '实例ID' }[curMode];
  document.getElementById('hud').textContent =
    '[' + modeStr + ']  显示: ' + vis.toLocaleString() +
    ' / 总占用: ' + total.toLocaleString() +
    '  |  网格 ' + NX + '×' + NY + '×' + NZ + ' (' + (NX * NY * NZ).toLocaleString() + ')';
}

/* =========================================================
   图例
   ========================================================= */
function mkLeg() {
  const el = document.getElementById('legend');

  if (curMode === 'sem') {
    let h = '<h4>语义类别 (点击隐藏)</h4>';
    for (let c = 0; c < 18; c++) {
      const info = CI[c]; if (!info) continue;
      const [r, g, b] = info.rgb;
      h += '<div class="lg' + (hidden.has(c) ? ' off' : '') + '" onclick="tog(' + c + ')">' +
        '<div class="ls" style="background:rgb(' + r + ',' + g + ',' + b + ')"></div>' +
        '<span class="lt">' + c + ' ' + info.zh + '</span></div>';
    }
    el.innerHTML = h;

  } else if (curMode === 'flow') {
    const maxSpd = curFlow.length ? Math.max(...curFlow.map(f => f[5]), 0) : 0;
    el.innerHTML = '<h4>Flow 速度 (m/s)</h4>' +
      '<div class="grad-wrap">' +
      '<div class="grad-strip"></div>' +
      '<div class="grad-labs"><span>0</span><span>' + (maxSpd / 2).toFixed(3) + '</span><span>' + maxSpd.toFixed(3) + '</span></div>' +
      '</div>' +
      '<div style="margin-top:6px;color:var(--dim)">动态体素: <b style="color:var(--accent)">' + curFlow.length + '</b></div>' +
      (maxSpd < 0.001 ? '<div style="color:#f87171;margin-top:3px;font-size:10px">⚠ 所有NPC静止 (speed≈0)</div>' : '');

  } else {
    // 实例模式：列出所有 instance ID
    const idMap = {};
    curVox.forEach(v => { const id = v[4]; idMap[id] = (idMap[id] || 0) + 1; });
    const sorted = Object.entries(idMap).sort((a, b) => b[1] - a[1]);
    let h = '<h4>实例 ID (' + sorted.length + ' 个)</h4>';
    sorted.forEach(([id, cnt]) => {
      const c = instClr(+id);
      h += '<div class="lg"><div class="ls" style="background:#' + c.getHexString() + '"></div>' +
        '<span class="lt">ID ' + id + '  <span style="color:var(--dim)">(' + cnt + ')</span></span></div>';
    });
    el.innerHTML = h;
  }
}

function tog(c) { hidden.has(c) ? hidden.delete(c) : hidden.add(c); mkLeg(); rebuild(); }

/* =========================================================
   模式切换
   ========================================================= */
function setMode(m) {
  curMode = m;
  ['sem', 'flow', 'inst'].forEach(k =>
    document.getElementById('btn-' + k).classList.toggle('active', k === m)
  );
  const titles = { sem: '统计 — 语义', flow: '统计 — Flow速度', inst: '统计 — 实例ID' };
  document.getElementById('statsTitle').textContent = titles[m];
  rebuild();
  fillStats(window._lastStats);
}

/* =========================================================
   数据获取
   ========================================================= */
async function init() {
  try {
    const d = await (await fetch('/api/frames')).json();
    frames = d.frames;
    document.getElementById('dirIn').value = d.data_dir;
    const sl = document.getElementById('slider');
    sl.max = Math.max(0, frames.length - 1);
    sl.value = 0; ci = 0;
    if (frames.length) loadFrame(0);
    else document.getElementById('ld').textContent = '目录中没有帧: ' + d.data_dir;
  } catch (e) {
    document.getElementById('ld').textContent = 'API 错误: ' + e.message;
  }
}

async function loadFrame(idx) {
  if (idx < 0 || idx >= frames.length) return;
  ci = idx;
  const fid = frames[idx];
  const fs = String(fid).padStart(6, '0');
  document.getElementById('flab').textContent = 'frame_' + fs + '  (' + (idx + 1) + '/' + frames.length + ')';
  document.getElementById('slider').value = idx;
  document.getElementById('ld').style.display = 'block';
  document.getElementById('ld').textContent = '加载 frame_' + fs + '...';

  /* 图像 */
  document.getElementById('imgL').src = '/img/left/frame_' + fs + '.png?' + Date.now();
  document.getElementById('imgR').src = '/img/right/frame_' + fs + '.png?' + Date.now();

  try {
    const d = await (await fetch('/api/voxel/' + fid)).json();
    curVox = d.voxels || [];
    curFlow = d.flow || [];
    window._lastStats = d.stats || {};
    rebuild();
    fillStats(d.stats || {});
    fillMeta(d.meta || {});
    document.getElementById('ld').style.display = 'none';
  } catch (e) {
    document.getElementById('ld').textContent = '加载错误: ' + e.message;
    console.error(e);
  }
}

/* =========================================================
   统计面板
   ========================================================= */
function fillStats(st) {
  const el = document.getElementById('statsBox');
  if (!st || !Object.keys(st).length) {
    el.innerHTML = '<div style="color:var(--dim)">无数据</div>'; return;
  }

  if (curMode === 'sem') {
    let h = '';
    for (let c = 0; c < 18; c++) {
      const n = st[String(c)] || 0; if (!n) continue;
      const info = CI[c];
      const [r, g, b] = info.rgb;
      const pct = (n / (st.total || 1) * 100).toFixed(1);
      h += '<div class="srow"><div class="sw" style="background:rgb(' + r + ',' + g + ',' + b + ')"></div>' +
        '<span class="sn">' + info.zh + '</span>' +
        '<span class="sc">' + n.toLocaleString() + ' (' + pct + '%)</span></div>';
    }
    const un = st['255'] || 0;
    if (un) {
      const pct = (un / (st.total || 1) * 100).toFixed(1);
      h += '<div class="srow"><div class="sw" style="background:#1e2732"></div>' +
        '<span class="sn">未观测</span><span class="sc">' + un.toLocaleString() + ' (' + pct + '%)</span></div>';
    }
    el.innerHTML = h || '<div style="color:var(--dim)">无占用体素</div>';

  } else if (curMode === 'flow') {
    if (!curFlow.length) {
      el.innerHTML = '<div style="color:var(--dim)">无动态体素（flow_mask全为0）</div>'; return;
    }
    const speeds = curFlow.map(f => f[5]);
    const maxSpd = Math.max(...speeds, 0);
    const meanSpd = speeds.reduce((a, b) => a + b, 0) / speeds.length;
    const stationary = speeds.filter(s => s < 0.01).length;
    el.innerHTML =
      '<div class="srow"><span class="sn">动态体素数</span><span class="sc">' + curFlow.length + '</span></div>' +
      '<div class="srow"><span class="sn">平均速度</span><span class="sc">' + meanSpd.toFixed(4) + ' m/s</span></div>' +
      '<div class="srow"><span class="sn">最大速度</span><span class="sc">' + maxSpd.toFixed(4) + ' m/s</span></div>' +
      '<div class="srow"><span class="sn">静止占比</span><span class="sc">' + (stationary / speeds.length * 100).toFixed(1) + '%</span></div>';

  } else {
    /* 实例模式统计 */
    const idMap = {};
    curVox.forEach(v => { const id = v[4]; idMap[id] = (idMap[id] || 0) + 1; });
    const sorted = Object.entries(idMap).sort((a, b) => b[1] - a[1]);
    let h = '<div class="srow"><span class="sn">总占用体素</span><span class="sc">' + curVox.length.toLocaleString() + '</span></div>';
    sorted.forEach(([id, cnt]) => {
      const c = instClr(+id);
      const pct = (cnt / curVox.length * 100).toFixed(1);
      h += '<div class="srow"><div class="sw" style="background:#' + c.getHexString() + '"></div>' +
        '<span class="sn">ID ' + id + '</span><span class="sc">' + cnt + ' (' + pct + '%)</span></div>';
    });
    el.innerHTML = h;
  }
}

/* =========================================================
   Meta 信息面板
   ========================================================= */
function fillMeta(m) {
  const el = document.getElementById('metaBox');
  if (!m || !Object.keys(m).length) {
    el.innerHTML = '<div style="color:var(--dim)">无 meta</div>'; return;
  }

  function row(k, v) {
    return '<div class="irow"><span class="ik">' + k + '</span><span class="iv">' + v + '</span></div>';
  }
  function divider() {
    return '<div class="divider"></div>';
  }

  let h = '';

  /* 相机信息 */
  const pos = m.camera_pos ? m.camera_pos.map(v => v.toFixed(2)).join(', ') : '—';
  const yaw = m.camera_yaw_rad != null ? (m.camera_yaw_rad * 180 / Math.PI).toFixed(1) + '°' : '—';
  h += row('相机位置', '(' + pos + ')');
  h += row('偏航角', yaw);
  h += row('相机高度', m.camera_height_m != null ? m.camera_height_m.toFixed(2) + ' m' : '—');

  /* 时间信息 */
  h += divider();
  h += row('时间戳', m.timestamp_sec != null ? m.timestamp_sec.toFixed(3) + ' s' : '—');
  h += row('仿真步数', m.sim_step != null ? m.sim_step : '—');

  /* NPC 信息 */
  h += divider();
  h += row('动态物体数', m.num_dynamic_objects != null ? m.num_dynamic_objects : '—');
  h += row('NPC数量', m.num_npc_positions != null ? m.num_npc_positions : '—');

  /* NPC 角速度 */
  if (m.npc_angular_velocities && m.npc_angular_velocities.length) {
    m.npc_angular_velocities.forEach((av, i) => {
      if (!av) return;
      const mag = Math.sqrt(av[0] ** 2 + av[1] ** 2 + av[2] ** 2);
      const label = i === 0 ? 'NPC0 |ω|' : 'NPC' + i + ' |ω|';
      const val = mag < 0.001 ? '0 rad/s (静止)' : mag.toFixed(4) + ' rad/s';
      h += row(label, val);
    });
  }

  /* Flow 统计 */
  if (m.flow_stats) {
    const fs = m.flow_stats;
    h += divider();
    h += row('动态体素', fs.dynamic_voxels + ' 个');
    if (fs.mean_speed_ms != null) h += row('平均速度', fs.mean_speed_ms.toFixed(4) + ' m/s');
    if (fs.max_speed_ms != null) h += row('最大速度', fs.max_speed_ms.toFixed(4) + ' m/s');
    if (fs.stationary_ratio != null) h += row('静止比例', (fs.stationary_ratio * 100).toFixed(1) + '%');
  }

  /* 体素统计 */
  if (m.voxel_stats) {
    const vs = m.voxel_stats;
    h += divider();
    h += row('体素总数', vs.total ? vs.total.toLocaleString() : '—');
    h += row('占用率', vs.total ? (vs.occupied / vs.total * 100).toFixed(1) + '%' : '—');
    h += row('未观测', vs.unobserved != null ? vs.unobserved : '—');
  }

  el.innerHTML = h;
}

/* =========================================================
   导航
   ========================================================= */
function go(delta, abs) {
  if (abs !== undefined) loadFrame(parseInt(abs));
  else loadFrame(ci + delta);
}

function chDir() {
  const v = document.getElementById('dirIn').value.trim();
  if (!v) return;
  fetch('/api/set_dir', { method: 'POST', body: v }).then(() => init());
}

document.addEventListener('keydown', function (e) {
  if (e.target.tagName === 'INPUT') return;  // 在输入框时不拦截
  if (e.key === 'ArrowLeft' || e.key === 'a') go(-1);
  if (e.key === 'ArrowRight' || e.key === 'd') go(+1);
  if (e.key === 'Home') go(0, 0);
  if (e.key === 'End') go(0, frames.length - 1);
  if (e.key === '1') setMode('sem');
  if (e.key === '2') setMode('flow');
  if (e.key === '3') setMode('inst');
});

/* =========================================================
   Resize + 渲染循环
   ========================================================= */
function onRz() {
  const vp = document.getElementById('vp');
  ren.setSize(vp.clientWidth, vp.clientHeight);
  cam.aspect = vp.clientWidth / vp.clientHeight;
  cam.updateProjectionMatrix();
}
window.addEventListener('resize', onRz);
if (typeof ResizeObserver !== 'undefined')
  new ResizeObserver(onRz).observe(document.getElementById('vp'));

(function loop() {
  requestAnimationFrame(loop);
  ctrl.update();
  ren.render(sc, cam);
})();

/* 启动 */
mkLeg(); onRz(); init();
</script></body></html>"""


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------
class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "VoxelViewer/3.0"

    def log_message(self, fmt, *a):
        msg = str(a[0] if a else "")
        if "/api/" in msg or "vendor" in msg:
            super().log_message(fmt, *a)

    def do_GET(self):
        path = self.path.split("?")[0]

        # 首页
        if path in ("/", ""):
            return self._html()

        # vendor JS 文件（Three.js + OrbitControls）
        m = re.match(r"/vendor/(.+\.js)$", path)
        if m:
            fp = os.path.join(VENDOR_DIR, m.group(1))
            return self._file(fp, "application/javascript")

        # API: 帧列表
        if path == "/api/frames":
            return self._json({"frames": scan_frames(DATA_DIR), "data_dir": DATA_DIR})

        # API: 体素数据（语义 + 实例 + Flow）
        m = re.match(r"/api/voxel/(\d+)$", path)
        if m:
            fid = int(m.group(1))
            d = load_voxels_json(DATA_DIR, fid)
            d["meta"] = load_meta(DATA_DIR, fid)
            return self._json(d)

        # 图片: /img/left/frame_000042.png
        m = re.match(r"/img/(left|right)/(.+\.png)$", path)
        if m:
            fp = os.path.join(DATA_DIR, m.group(1), m.group(2))
            return self._file(fp, "image/png")

        self.send_error(404)

    def do_POST(self):
        global DATA_DIR
        if self.path.split("?")[0] == "/api/set_dir":
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n).decode().strip()
            if os.path.isdir(body):
                DATA_DIR = os.path.abspath(body)
                print(f"[Viewer] Dir -> {DATA_DIR}")
            return self._json({"ok": True, "data_dir": DATA_DIR})
        self.send_error(404)

    def do_HEAD(self):
        path = self.path.split("?")[0]
        m = re.match(r"/img/(left|right)/(.+\.png)$", path)
        if m:
            fp = os.path.join(DATA_DIR, m.group(1), m.group(2))
            if os.path.isfile(fp):
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", os.path.getsize(fp))
                self.end_headers()
                return
        self.send_error(404)

    def _html(self):
        page = FRONTEND.replace("__CLASS_JSON__", class_info_js())
        raw = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html;charset=utf-8")
        self.send_header("Content-Length", len(raw))
        self.end_headers()
        self.wfile.write(raw)

    def _json(self, obj):
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json;charset=utf-8")
        self.send_header("Content-Length", len(raw))
        self.end_headers()
        self.wfile.write(raw)

    def _file(self, fp, mime):
        if not os.path.isfile(fp):
            return self.send_error(404, f"Not found: {fp}")
        with open(fp, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", len(data))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    for fname in ["three.min.js", "OrbitControls.js"]:
        fp = os.path.join(VENDOR_DIR, fname)
        if not os.path.isfile(fp):
            print(f"[Viewer] ERROR: Missing {fp}")
            print(f"         Please download Three.js r137 files to {VENDOR_DIR}/")
            sys.exit(1)

    port = find_free_port(args.port)
    if port is None:
        print(f"[Viewer] ERROR: No free port found near {args.port}")
        sys.exit(1)
    if port != args.port:
        print(f"[Viewer] Port {args.port} busy, using {port}")

    srv = http.server.HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"

    nf = len(scan_frames(DATA_DIR))
    print(f"[Viewer] {nf} frames in {DATA_DIR}")
    print(f"[Viewer] Open {url}")
    print(f"[Viewer] Ctrl+C to quit")

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[Viewer] Bye.")
        srv.shutdown()


if __name__ == "__main__":
    main()
