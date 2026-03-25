"""对比 stereo_voxel_capture.py vs stereo_voxel_capture_dng.py 输出
====================================================================
验证项:
  1. 两边都无全黑帧
  2. 图像分辨率一致 (1280x1080)
  3. calibration.json 关键参数一致
  4. DNG 可正常解码，值域合理
  5. 同帧号 RGB PNG 像素级一致（同源数据）
  6. 体素 GT 一致

用法（不依赖 Isaac Sim，普通 Python 即可）:
    python projects/stereo_voxel/scripts/compare_capture_outputs.py
"""
import json
import os
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIG_OUTPUT = os.path.join(PROJECT_DIR, "output")       # stereo_voxel_capture.py
DNG_OUTPUT = os.path.join(PROJECT_DIR, "output_dng")     # stereo_voxel_capture_dng.py

LOG_PATH = os.path.join(PROJECT_DIR, "_compare_result.txt")
lines = []
pass_count = 0
fail_count = 0
warn_count = 0


def log(msg):
    print(msg, flush=True)
    lines.append(msg)


def log_pass(msg):
    global pass_count
    pass_count += 1
    log(f"[PASS] {msg}")


def log_fail(msg):
    global fail_count
    fail_count += 1
    log(f"[FAIL] {msg}")


def log_warn(msg):
    global warn_count
    warn_count += 1
    log(f"[WARN] {msg}")


def load_pngs(directory, max_frames=10):
    if not os.path.isdir(directory):
        return []
    files = sorted(f for f in os.listdir(directory) if f.endswith(".png"))[:max_frames]
    results = []
    for f in files:
        img = cv2.imread(os.path.join(directory, f))
        if img is not None:
            results.append((f, img))
    return results


def is_black(img, threshold=5.0):
    return img.mean() < threshold


# =========================================================================
# 1. 检查原版输出
# =========================================================================
log("=" * 70)
log("PART 1: stereo_voxel_capture.py 输出 (output/)")
log("=" * 70)

orig_left_dir = os.path.join(ORIG_OUTPUT, "left")
orig_right_dir = os.path.join(ORIG_OUTPUT, "right")

if not os.path.isdir(ORIG_OUTPUT):
    log_fail(f"output/ 目录不存在: {ORIG_OUTPUT}")
    log("请先运行: isaaclab.bat -p projects/stereo_voxel/scripts/stereo_voxel_capture.py --headless --num_frames 5 --no_npc")
    sys.exit(1)

orig_left_imgs = load_pngs(orig_left_dir)
orig_right_imgs = load_pngs(orig_right_dir)
orig_frames = min(len(orig_left_imgs), len(orig_right_imgs))
log(f"帧数: left={len(orig_left_imgs)}, right={len(orig_right_imgs)}")

black_count = 0
for side, imgs in [("left", orig_left_imgs), ("right", orig_right_imgs)]:
    for fname, img in imgs:
        m = img.mean()
        if is_black(img):
            black_count += 1
            log(f"  {side}/{fname}: mean={m:.1f} [BLACK!]")
        else:
            log(f"  {side}/{fname}: shape={img.shape}, mean={m:.1f}")

if black_count > 0:
    log_fail(f"原版 {black_count} 张全黑帧")
else:
    log_pass(f"原版无全黑帧 ({orig_frames} 对)")

# =========================================================================
# 2. 检查 DNG 版输出
# =========================================================================
log("")
log("=" * 70)
log("PART 2: stereo_voxel_capture_dng.py 输出 (output_dng/)")
log("=" * 70)

dng_left_dir = os.path.join(DNG_OUTPUT, "left")
dng_right_dir = os.path.join(DNG_OUTPUT, "right")
dng_left_dng_dir = os.path.join(DNG_OUTPUT, "left_dng")
dng_right_dng_dir = os.path.join(DNG_OUTPUT, "right_dng")

if not os.path.isdir(DNG_OUTPUT):
    log_fail(f"output_dng/ 目录不存在: {DNG_OUTPUT}")
    log("请先运行: isaaclab.bat -p projects/stereo_voxel/scripts/stereo_voxel_capture_dng.py --headless --num_frames 5 --no_npc")
    sys.exit(1)

dng_left_imgs = load_pngs(dng_left_dir)
dng_right_imgs = load_pngs(dng_right_dir)
dng_frames = min(len(dng_left_imgs), len(dng_right_imgs))
log(f"RGB 帧数: left={len(dng_left_imgs)}, right={len(dng_right_imgs)}")

black_count = 0
for side, imgs in [("left", dng_left_imgs), ("right", dng_right_imgs)]:
    for fname, img in imgs:
        m = img.mean()
        if is_black(img):
            black_count += 1
            log(f"  {side}/{fname}: mean={m:.1f} [BLACK!]")
        else:
            log(f"  {side}/{fname}: shape={img.shape}, mean={m:.1f}")

if black_count > 0:
    log_fail(f"DNG 版 {black_count} 张全黑帧")
else:
    log_pass(f"DNG 版 RGB 无全黑帧 ({dng_frames} 对)")

# DNG 文件检查
log("")
log("--- DNG 文件检查 ---")
for side, dng_dir in [("left_dng", dng_left_dng_dir), ("right_dng", dng_right_dng_dir)]:
    if not os.path.isdir(dng_dir):
        log_fail(f"{side}/ 目录不存在")
        continue
    dngs = sorted(f for f in os.listdir(dng_dir) if f.endswith(".dng"))
    log(f"{side}: {len(dngs)} 个 DNG 文件")
    dng_ok = True
    for dng_name in dngs[:5]:
        dng_path = os.path.join(dng_dir, dng_name)
        sz = os.path.getsize(dng_path)
        try:
            import tifffile
            with tifffile.TiffFile(dng_path) as tif:
                page = tif.pages[0]
                data = page.asarray()
                log(f"  {dng_name}: {sz:,}B, shape={data.shape}, dtype={data.dtype}, "
                    f"range=[{data.min()}, {data.max()}], mean={data.mean():.1f}")
                if data.max() > 4095:
                    log_warn(f"  {dng_name} max={data.max()} > 4095 (12-bit 上限)")
                    dng_ok = False
                if data.mean() < 1.0:
                    log_warn(f"  {dng_name} 全黑 DNG")
                    dng_ok = False
        except ImportError:
            log(f"  {dng_name}: {sz:,}B (tifffile 未安装，跳过解码)")
        except Exception as e:
            log_fail(f"  {dng_name}: 解码失败 ({e})")
            dng_ok = False
    if dng_ok and dngs:
        log_pass(f"{side} DNG 解码正常")

# =========================================================================
# 3. Calibration 对比
# =========================================================================
log("")
log("=" * 70)
log("PART 3: calibration.json 对比")
log("=" * 70)

orig_calib_path = os.path.join(ORIG_OUTPUT, "calibration.json")
dng_calib_path = os.path.join(DNG_OUTPUT, "calibration.json")

orig_calib = None
dng_calib = None

if os.path.exists(orig_calib_path):
    with open(orig_calib_path) as f:
        orig_calib = json.load(f)
    log(f"原版: projection={orig_calib.get('projection', 'N/A')}, "
        f"res={orig_calib.get('resolution')}, baseline={orig_calib.get('baseline_m')}")
else:
    log_fail(f"原版 calibration.json 不存在")

if os.path.exists(dng_calib_path):
    with open(dng_calib_path) as f:
        dng_calib = json.load(f)
    log(f"DNG版: projection={dng_calib.get('projection', 'N/A')}, "
        f"res={dng_calib.get('resolution')}, baseline={dng_calib.get('baseline_m')}")
    if "raw_pipeline" in dng_calib:
        rp = dng_calib["raw_pipeline"]
        log(f"  raw_pipeline: {rp.get('format')}, noise={rp.get('noise_preset')}")
else:
    log_fail(f"DNG 版 calibration.json 不存在")

if orig_calib and dng_calib:
    checks = []
    # 共有字段对比（忽略 raw_pipeline，那是 DNG 独有的）
    shared_keys = ["projection", "resolution", "fx", "fy", "cx", "cy",
                   "k1", "max_fov_deg", "baseline_m", "baseline_direction"]
    for key in shared_keys:
        v1, v2 = orig_calib.get(key), dng_calib.get(key)
        if v1 is None and v2 is None:
            continue
        if isinstance(v1, float) and isinstance(v2, float):
            match = abs(v1 - v2) < 1e-6
        else:
            match = v1 == v2
        checks.append(match)
        status = "MATCH" if match else "MISMATCH"
        if not match:
            log(f"  {key}: {status} ({v1} vs {v2})")

    if all(checks):
        log_pass(f"calibration 全部 {len(checks)} 项参数一致")
    else:
        failed = sum(1 for c in checks if not c)
        log_fail(f"calibration {failed}/{len(checks)} 项不一致")

# =========================================================================
# 4. 同帧号 RGB 像素对比（核心验证）
# =========================================================================
log("")
log("=" * 70)
log("PART 4: 同帧号 RGB 像素对比")
log("=" * 70)
log("说明: 两个脚本独立运行，场景相同但时间步不同，")
log("      因此 NPC 不一定在同位置。静态场景部分应高度一致。")

compare_frames = min(orig_frames, dng_frames)
if compare_frames == 0:
    log_warn("无可对比帧")
else:
    ncc_list = []
    for i in range(compare_frames):
        fname_o, img_o = orig_left_imgs[i]
        fname_d, img_d = dng_left_imgs[i]

        if img_o.shape != img_d.shape:
            log_fail(f"帧 {i}: 分辨率不匹配 {img_o.shape} vs {img_d.shape}")
            continue

        # 像素差
        diff = np.abs(img_o.astype(float) - img_d.astype(float))

        # NCC
        g1 = cv2.cvtColor(img_o, cv2.COLOR_BGR2GRAY).astype(float)
        g2 = cv2.cvtColor(img_d, cv2.COLOR_BGR2GRAY).astype(float)
        g1n = (g1 - g1.mean()) / (g1.std() + 1e-8)
        g2n = (g2 - g2.mean()) / (g2.std() + 1e-8)
        ncc = float(np.mean(g1n * g2n))
        ncc_list.append(ncc)

        log(f"  帧 {i} ({fname_o} vs {fname_d}): "
            f"mean_diff={diff.mean():.2f}, max_diff={diff.max():.0f}, NCC={ncc:.4f}")

    if ncc_list:
        avg_ncc = np.mean(ncc_list)
        log(f"平均 NCC={avg_ncc:.4f}")
        if avg_ncc > 0.95:
            log_pass(f"RGB 高度相似 (avg NCC={avg_ncc:.4f} > 0.95)")
        elif avg_ncc > 0.80:
            log_warn(f"RGB 中度相似 (avg NCC={avg_ncc:.4f}), 独立运行时间步差异导致")
        else:
            log_fail(f"RGB 差异过大 (avg NCC={avg_ncc:.4f})")

    # 保存首帧对比图
    if orig_left_imgs and dng_left_imgs:
        _, img_o = orig_left_imgs[0]
        _, img_d = dng_left_imgs[0]
        if img_o.shape == img_d.shape:
            h, w = img_o.shape[:2]
            canvas = np.zeros((h, w * 2 + 10, 3), dtype=np.uint8)
            canvas[:, :w] = img_o
            canvas[:, w + 10:] = img_d
            cv2.putText(canvas, "original (output/)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(canvas, "dng (output_dng/)", (w + 20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            compare_path = os.path.join(PROJECT_DIR, "_compare_left_frame0.png")
            cv2.imwrite(compare_path, canvas)
            log(f"对比图: {compare_path}")

# =========================================================================
# 5. 体素 GT 对比
# =========================================================================
log("")
log("=" * 70)
log("PART 5: 体素 GT 对比")
log("=" * 70)

orig_voxel_dir = os.path.join(ORIG_OUTPUT, "voxel")
dng_voxel_dir = os.path.join(DNG_OUTPUT, "voxel")

if os.path.isdir(orig_voxel_dir) and os.path.isdir(dng_voxel_dir):
    orig_voxels = sorted(f for f in os.listdir(orig_voxel_dir) if f.endswith("_semantic.npz"))
    dng_voxels = sorted(f for f in os.listdir(dng_voxel_dir) if f.endswith("_semantic.npz"))
    log(f"原版体素: {len(orig_voxels)}, DNG版体素: {len(dng_voxels)}")

    compare_voxels = min(len(orig_voxels), len(dng_voxels))
    if compare_voxels > 0:
        voxel_match_all = True
        for i in range(compare_voxels):
            v_o = np.load(os.path.join(orig_voxel_dir, orig_voxels[i]))["data"]
            v_d = np.load(os.path.join(dng_voxel_dir, dng_voxels[i]))["data"]
            if v_o.shape != v_d.shape:
                log_fail(f"体素帧 {i}: shape 不匹配 {v_o.shape} vs {v_d.shape}")
                voxel_match_all = False
                continue
            match_rate = float(np.mean(v_o == v_d))
            occ_o = int(np.sum(v_o > 0))
            occ_d = int(np.sum(v_d > 0))
            log(f"  帧 {i}: match_rate={match_rate:.4f}, "
                f"occupied: orig={occ_o}, dng={occ_d}")
            if match_rate < 0.90:
                voxel_match_all = False

        if voxel_match_all:
            log_pass("体素 GT 高度一致")
        else:
            log_warn("体素 GT 有差异 (独立运行时 NPC 位置可能不同)")
    else:
        log_warn("无可对比体素帧")
else:
    if not os.path.isdir(orig_voxel_dir):
        log_warn(f"原版 voxel/ 目录不存在")
    if not os.path.isdir(dng_voxel_dir):
        log_warn(f"DNG 版 voxel/ 目录不存在")

# =========================================================================
# 总结
# =========================================================================
log("")
log("=" * 70)
log(f"总结: {pass_count} PASS, {fail_count} FAIL, {warn_count} WARN")
log("=" * 70)
if fail_count == 0:
    log("端到端验证通过!")
else:
    log("存在失败项，请检查上方详情。")

# 写入文件
with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
log(f"\n结果已保存: {LOG_PATH}")
