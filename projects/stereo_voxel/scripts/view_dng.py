"""DNG + RGB 并列可视化浏览器 (浏览器版)
============================================
生成交互式 HTML，在浏览器中查看 DNG vs RGB 对比。

用法:
  isaaclab.bat -p projects/stereo_voxel/scripts/view_dng.py
  isaaclab.bat -p projects/stereo_voxel/scripts/view_dng.py --dng_dir path/to/dng --rgb_dir path/to/rgb
"""
import argparse
import base64
import io
import json
import os
import sys
import webbrowser

import numpy as np

try:
    import tifffile
except ImportError:
    print("需要 tifffile: isaaclab.bat -p -m pip install tifffile")
    sys.exit(1)

from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="DNG + RGB 并列可视化")
    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--dng_dir", default=os.path.join(project, "output_dng", "left_dng"))
    p.add_argument("--rgb_dir", default=os.path.join(project, "output_dng", "left"))
    p.add_argument("--rgb_orig_dir", default=os.path.join(project, "output", "left"))
    p.add_argument("--right_dng_dir", default=os.path.join(project, "output_dng", "right_dng"))
    p.add_argument("--right_rgb_dir", default=os.path.join(project, "output_dng", "right"))
    p.add_argument("--right_rgb_orig_dir", default=os.path.join(project, "output", "right"))
    return p.parse_args()


# =========================================================================
# DNG 解码
# =========================================================================
def load_dng(path):
    with tifffile.TiffFile(path) as tif:
        return tif.pages[0].asarray()


def demosaic_half(bayer, pattern="RGGB"):
    """半分辨率去马赛克: 2x2 → 1 pixel"""
    f = bayer.astype(np.float32)
    if pattern == "RGGB":
        r = f[0::2, 0::2]
        g = (f[0::2, 1::2] + f[1::2, 0::2]) / 2.0
        b = f[1::2, 1::2]
    else:
        r = f[0::2, 0::2]
        g = (f[0::2, 1::2] + f[1::2, 0::2]) / 2.0
        b = f[1::2, 1::2]
    mx = max(r.max(), g.max(), b.max(), 1.0)
    rgb = np.stack([r, g, b], axis=-1) / mx * 255.0
    return rgb.clip(0, 255).astype(np.uint8)


def bayer_pseudocolor(bayer, pattern="RGGB"):
    h, w = bayer.shape
    mx = max(float(bayer.max()), 1.0)
    f = bayer.astype(np.float32) / mx * 255.0
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    if pattern == "RGGB":
        rgb[0::2, 0::2, 0] = f[0::2, 0::2]
        rgb[0::2, 1::2, 1] = f[0::2, 1::2]
        rgb[1::2, 0::2, 1] = f[1::2, 0::2]
        rgb[1::2, 1::2, 2] = f[1::2, 1::2]
    return rgb


def bayer_gray(bayer):
    mx = max(float(bayer.max()), 1.0)
    g = (bayer.astype(np.float32) / mx * 255).clip(0, 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def numpy_to_b64png(arr):
    """numpy RGB array → base64 PNG string"""
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def file_to_b64png(path):
    """PNG file → base64 string"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# =========================================================================
# 生成数据
# =========================================================================
def process_frames(args):
    dng_files = sorted(f for f in os.listdir(args.dng_dir) if f.endswith(".dng"))
    rgb_files = sorted(f for f in os.listdir(args.rgb_dir) if f.endswith(".png"))
    dng_stems = {os.path.splitext(f)[0]: f for f in dng_files}
    rgb_stems = {os.path.splitext(f)[0]: f for f in rgb_files}
    common = sorted(set(dng_stems) & set(rgb_stems))

    # 右眼
    r_dng_stems, r_rgb_stems = {}, {}
    if os.path.isdir(args.right_dng_dir):
        r_dng_files = sorted(f for f in os.listdir(args.right_dng_dir) if f.endswith(".dng"))
        r_dng_stems = {os.path.splitext(f)[0]: f for f in r_dng_files}
    if os.path.isdir(args.right_rgb_dir):
        r_rgb_files = sorted(f for f in os.listdir(args.right_rgb_dir) if f.endswith(".png"))
        r_rgb_stems = {os.path.splitext(f)[0]: f for f in r_rgb_files}

    orig_rgb_stems, r_orig_rgb_stems = {}, {}
    if os.path.isdir(args.rgb_orig_dir):
        orig_files = sorted(f for f in os.listdir(args.rgb_orig_dir) if f.endswith(".png"))
        orig_rgb_stems = {os.path.splitext(f)[0]: f for f in orig_files}
    if os.path.isdir(args.right_rgb_orig_dir):
        r_orig_files = sorted(f for f in os.listdir(args.right_rgb_orig_dir) if f.endswith(".png"))
        r_orig_rgb_stems = {os.path.splitext(f)[0]: f for f in r_orig_files}

    if not common:
        print(f"无匹配帧! DNG: {len(dng_files)}, RGB: {len(rgb_files)}")
        sys.exit(1)

    print(f"处理 {len(common)} 帧...")
    frames = []

    for i, stem in enumerate(common):
        print(f"  [{i+1}/{len(common)}] {stem}", end="", flush=True)
        frame = {"name": stem}

        # 左眼 DNG
        dng_path = os.path.join(args.dng_dir, dng_stems[stem])
        bayer = load_dng(dng_path)

        frame["left_demosaic"] = numpy_to_b64png(demosaic_half(bayer))
        frame["left_bayer"] = numpy_to_b64png(bayer_pseudocolor(bayer))
        frame["left_gray"] = numpy_to_b64png(bayer_gray(bayer))
        frame["left_info"] = f"{bayer.dtype} [{bayer.min()}-{bayer.max()}] mean={bayer.mean():.0f}"

        # 左眼 RGB
        rgb_path = os.path.join(args.rgb_dir, rgb_stems[stem])
        frame["left_rgb"] = file_to_b64png(rgb_path)
        rgb_arr = np.array(Image.open(rgb_path))
        frame["left_rgb_info"] = f"mean={rgb_arr.mean():.1f}"

        # 左眼原版 RGB
        if stem in orig_rgb_stems:
            orig_path = os.path.join(args.rgb_orig_dir, orig_rgb_stems[stem])
            frame["left_orig"] = file_to_b64png(orig_path)
            orig_arr = np.array(Image.open(orig_path))
            frame["left_orig_info"] = f"mean={orig_arr.mean():.1f}"

        # 右眼
        if stem in r_dng_stems and stem in r_rgb_stems:
            r_dng_path = os.path.join(args.right_dng_dir, r_dng_stems[stem])
            r_bayer = load_dng(r_dng_path)
            frame["right_demosaic"] = numpy_to_b64png(demosaic_half(r_bayer))
            frame["right_bayer"] = numpy_to_b64png(bayer_pseudocolor(r_bayer))
            frame["right_gray"] = numpy_to_b64png(bayer_gray(r_bayer))
            frame["right_info"] = f"{r_bayer.dtype} [{r_bayer.min()}-{r_bayer.max()}] mean={r_bayer.mean():.0f}"

            r_rgb_path = os.path.join(args.right_rgb_dir, r_rgb_stems[stem])
            frame["right_rgb"] = file_to_b64png(r_rgb_path)
            r_rgb_arr = np.array(Image.open(r_rgb_path))
            frame["right_rgb_info"] = f"mean={r_rgb_arr.mean():.1f}"

            if stem in r_orig_rgb_stems:
                r_orig_path = os.path.join(args.right_rgb_orig_dir, r_orig_rgb_stems[stem])
                frame["right_orig"] = file_to_b64png(r_orig_path)
                r_orig_arr = np.array(Image.open(r_orig_path))
                frame["right_orig_info"] = f"mean={r_orig_arr.mean():.1f}"

        frames.append(frame)
        print(" OK")

    return frames


# =========================================================================
# HTML 模板
# =========================================================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>DNG Viewer</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #1a1a2e; color: #eee; font-family: 'Segoe UI', sans-serif; }
.header { background: #16213e; padding: 10px 20px; display: flex; align-items: center; gap: 15px; }
.header h2 { font-size: 16px; color: #0f3460; background: #e94560; padding: 4px 12px; border-radius: 4px; }
.controls { display: flex; gap: 8px; align-items: center; }
.controls button {
    background: #0f3460; color: #eee; border: none; padding: 6px 14px;
    border-radius: 4px; cursor: pointer; font-size: 13px;
}
.controls button:hover { background: #e94560; }
.controls button.active { background: #e94560; }
.controls .sep { color: #555; margin: 0 5px; }
.info { color: #888; font-size: 12px; margin-left: auto; }
.grid {
    display: grid; grid-template-columns: 1fr 1fr 1fr;
    gap: 4px; padding: 8px; height: calc(100vh - 52px);
}
.grid.two-col { grid-template-columns: 1fr 1fr; }
.panel { position: relative; background: #0a0a1a; border-radius: 4px; overflow: hidden; }
.panel img {
    width: 100%; height: 100%; object-fit: contain; display: block;
}
.panel .label {
    position: absolute; top: 6px; left: 8px;
    background: rgba(0,0,0,0.7); color: #0f0; padding: 2px 8px;
    border-radius: 3px; font-size: 12px;
}
.panel .stat {
    position: absolute; bottom: 6px; left: 8px;
    background: rgba(0,0,0,0.7); color: #aaa; padding: 2px 8px;
    border-radius: 3px; font-size: 11px;
}
kbd { background: #333; padding: 1px 5px; border-radius: 3px; font-size: 11px; }
</style>
</head><body>
<div class="header">
    <h2>DNG Viewer</h2>
    <div class="controls">
        <button onclick="go(-1)" title="← A">◀ Prev</button>
        <button onclick="go(1)" title="→ D Space">Next ▶</button>
        <span class="sep">|</span>
        <button id="btn-eye-l" class="active" onclick="setEye('left')">L eye</button>
        <button id="btn-eye-r" onclick="setEye('right')">R eye</button>
        <span class="sep">|</span>
        <button id="btn-m1" class="active" onclick="setMode(1)">1 Demosaic</button>
        <button id="btn-m2" onclick="setMode(2)">2 Bayer</button>
        <button id="btn-m3" onclick="setMode(3)">3 Gray</button>
    </div>
    <div class="info">
        <span id="frame-info"></span>
        &nbsp; <kbd>←→</kbd> 翻页 <kbd>L</kbd><kbd>R</kbd> 切眼 <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd> 模式
    </div>
</div>
<div class="grid" id="grid">
    <div class="panel">
        <div class="label" id="lbl-dng">DNG (demosaic)</div>
        <img id="img-dng">
        <div class="stat" id="stat-dng"></div>
    </div>
    <div class="panel">
        <div class="label">RGB (DNG ver)</div>
        <img id="img-rgb">
        <div class="stat" id="stat-rgb"></div>
    </div>
    <div class="panel" id="panel-orig">
        <div class="label">RGB (original)</div>
        <img id="img-orig">
        <div class="stat" id="stat-orig"></div>
    </div>
</div>
<script>
const FRAMES = __FRAMES_JSON__;
let idx = 0, eye = 'left', mode = 1;
const modeKeys = {1: 'demosaic', 2: 'bayer', 3: 'gray'};

function update() {
    const f = FRAMES[idx];
    const prefix = eye + '_';
    const dngKey = prefix + modeKeys[mode];
    const dngSrc = f[dngKey];
    document.getElementById('img-dng').src = dngSrc ? 'data:image/png;base64,' + dngSrc : '';
    document.getElementById('stat-dng').textContent = f[prefix + 'info'] || '';
    document.getElementById('lbl-dng').textContent = 'DNG (' + modeKeys[mode] + ')';

    const rgbSrc = f[prefix + 'rgb'];
    document.getElementById('img-rgb').src = rgbSrc ? 'data:image/png;base64,' + rgbSrc : '';
    document.getElementById('stat-rgb').textContent = f[prefix + 'rgb_info'] || '';

    const origSrc = f[prefix + 'orig'];
    const origPanel = document.getElementById('panel-orig');
    const grid = document.getElementById('grid');
    if (origSrc) {
        document.getElementById('img-orig').src = 'data:image/png;base64,' + origSrc;
        document.getElementById('stat-orig').textContent = f[prefix + 'orig_info'] || '';
        origPanel.style.display = '';
        grid.classList.remove('two-col');
    } else {
        origPanel.style.display = 'none';
        grid.classList.add('two-col');
    }

    document.getElementById('frame-info').textContent =
        '[' + (idx+1) + '/' + FRAMES.length + '] ' + f.name + ' | ' + eye + ' eye';

    // button highlights
    ['btn-m1','btn-m2','btn-m3'].forEach((id,i) => {
        document.getElementById(id).classList.toggle('active', mode === i+1);
    });
    document.getElementById('btn-eye-l').classList.toggle('active', eye === 'left');
    document.getElementById('btn-eye-r').classList.toggle('active', eye === 'right');
}

function go(d) { idx = (idx + d + FRAMES.length) % FRAMES.length; update(); }
function setEye(e) { eye = e; update(); }
function setMode(m) { mode = m; update(); }

document.addEventListener('keydown', e => {
    if (e.key === 'ArrowRight' || e.key === 'd' || e.key === ' ') go(1);
    else if (e.key === 'ArrowLeft' || e.key === 'a') go(-1);
    else if (e.key === '1') setMode(1);
    else if (e.key === '2') setMode(2);
    else if (e.key === '3') setMode(3);
    else if (e.key === 'l') setEye('left');
    else if (e.key === 'r') setEye('right');
});

update();
</script>
</body></html>"""


def main():
    args = parse_args()
    frames = process_frames(args)

    # 生成 HTML
    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(project, "view_dng.html")

    frames_json = json.dumps(frames, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("__FRAMES_JSON__", frames_json)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\n生成: {out_path} ({size_mb:.1f} MB)")
    print("正在打开浏览器...")
    webbrowser.open(f"file:///{out_path.replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
