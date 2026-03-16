"""
Mixamo 角色巡逻演示
====================
架构：
  • Wrapper Xform（PatrolAgent 每帧更新位置 + 朝向）
    └── Char Xform（Mixamo USD 引用，UsdSkel 动画随 timeline 自动播放）

为什么这样可行：
  Mixamo 导出的 USD 包含直接的 UsdSkel 时间采样数据（SkelAnimation prim）。
  timeline.play() 后，stage 时间自动推进，骨骼关键帧插值 → 角色行走动画。
  不依赖 omni.anim.graph.core 状态机（Isaac People 的方案在 Standalone 失败的原因）。

前提：
  已按 docs/npc.md 流程准备好 Mixamo USD 文件。

运行：
    isaaclab.bat -p scripts/npc/npc_mixamo_patrol.py
    isaaclab.bat -p scripts/npc/npc_mixamo_patrol.py --headless
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Mixamo NPC patrol with UsdSkel animation")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import time
import numpy as np
import carb
import omni.usd
import omni.timeline
from pxr import Gf, Usd, UsdGeom, UsdSkel, UsdLux

# ── 路径常量 ──────────────────────────────────────────────────────────────
ASSET_ROOT = "D:/code/IsaacLab/Assets"
SCENE_USD  = f"{ASSET_ROOT}/Isaac/5.1/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
CHAR_ROOT  = "/World/Characters"

# ── NPC 配置（替换 usd 路径为实际转换好的 Mixamo USD）────────────────────
# 每个 NPC 可以用不同的角色 USD
MIXAMO_CHAR_USD = f"{ASSET_ROOT}/Custom/Characters/worker_walk/worker_walk.usd"

NPC_CONFIGS = [
    {
        "name":  "Worker_01",
        "usd":   MIXAMO_CHAR_USD,
        "start": np.array([ 9.0, 0.0,  2.0]),
        "end":   np.array([-9.0, 0.0,  2.0]),
        "speed": 1.2,
    },
    {
        "name":  "Worker_02",
        "usd":   MIXAMO_CHAR_USD,
        "start": np.array([ 9.0, 0.0,  0.0]),
        "end":   np.array([-9.0, 0.0,  0.0]),
        "speed": 1.0,
    },
    {
        "name":  "Worker_03",
        "usd":   MIXAMO_CHAR_USD,
        "start": np.array([ 9.0, 0.0, -2.0]),
        "end":   np.array([-9.0, 0.0, -2.0]),
        "speed": 1.4,
    },
]

FPS = 60.0  # 仿真帧率


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数：从 USD 获取动画时长（用于设置循环范围）
# ─────────────────────────────────────────────────────────────────────────────
def get_anim_duration(usd_path: str) -> tuple[float, float, float]:
    """
    返回 (start_time, end_time, fps)。
    如果 USD 没有动画数据，返回 (0, 120, 30)（默认 4 秒）。
    """
    try:
        stage = Usd.Stage.Open(usd_path)
        start = stage.GetStartTimeCode()
        end   = stage.GetEndTimeCode()
        fps   = stage.GetFramesPerSecond()
        if end > start:
            return (start, end, fps)
    except Exception as e:
        carb.log_warn(f"[Mixamo] 无法读取动画时长：{e}")
    return (0.0, 120.0, 30.0)


# ─────────────────────────────────────────────────────────────────────────────
# 巡逻控制器（只操作 Wrapper，不触及 Char 子节点）
# ─────────────────────────────────────────────────────────────────────────────
class PatrolAgent:
    """
    每帧更新 Wrapper Xform 的位置和朝向。
    骨骼动画由 timeline 自动驱动，本类不负责动画。
    """

    def __init__(self, cfg: dict):
        self.name  = cfg["name"]
        self.start = cfg["start"].copy()
        self.end   = cfg["end"].copy()
        self.speed = cfg["speed"]
        self.pos   = cfg["start"].copy().astype(float)
        self.going_forward = True
        self.wrapper_path  = f"{CHAR_ROOT}/{self.name}"

        # 朝向角（Y 轴旋转）：角色面朝行进方向
        d = self.end - self.start
        self._yaw_fwd = float(np.degrees(np.arctan2(d[0], d[2])))
        self._yaw_bwd = (self._yaw_fwd + 180.0) % 360.0

        self._translate_op = None
        self._rotate_op    = None

    def _init_ops(self, stage) -> bool:
        if self._translate_op is not None:
            return True
        prim = stage.GetPrimAtPath(self.wrapper_path)
        if not prim.IsValid():
            return False
        xform = UsdGeom.Xformable(prim)
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                self._translate_op = op
            elif op.GetOpType() == UsdGeom.XformOp.TypeRotateY:
                self._rotate_op = op
        if self._translate_op is None:
            xform.ClearXformOpOrder()
            self._translate_op = xform.AddTranslateOp()
            self._translate_op.Set(Gf.Vec3d(*self.pos.tolist()))
            self._rotate_op = xform.AddRotateYOp()
            self._rotate_op.Set(self._yaw_fwd)
        return True

    def step(self, stage, dt: float) -> np.ndarray:
        if not self._init_ops(stage):
            return self.pos.copy()

        target    = self.end if self.going_forward else self.start
        direction = target - self.pos
        dist      = np.linalg.norm(direction)

        if dist < 0.05:
            self.going_forward = not self.going_forward
            target    = self.end if self.going_forward else self.start
            direction = target - self.pos
            dist      = np.linalg.norm(direction) + 1e-6

        step_dist = self.speed * dt
        self.pos  = target.copy() if step_dist >= dist else self.pos + direction / dist * step_dist

        self._translate_op.Set(Gf.Vec3d(float(self.pos[0]),
                                         float(self.pos[1]),
                                         float(self.pos[2])))
        yaw = self._yaw_fwd if self.going_forward else self._yaw_bwd
        if self._rotate_op:
            self._rotate_op.Set(float(yaw))

        return self.pos.copy()


# ─────────────────────────────────────────────────────────────────────────────
# 场景和角色
# ─────────────────────────────────────────────────────────────────────────────
def load_scene() -> "Usd.Stage":
    ctx = omni.usd.get_context()
    carb.log_info(f"[Mixamo] Opening scene: {SCENE_USD}")
    ctx.open_stage(SCENE_USD)
    for _ in range(400):
        simulation_app.update()
        if ctx.get_stage().GetPrimAtPath("/World").IsValid():
            break
        time.sleep(0.05)
    return ctx.get_stage()


def setup_lighting(stage):
    if not stage.GetPrimAtPath("/World/_Sun_").IsValid():
        sun = UsdLux.DistantLight.Define(stage, "/World/_Sun_")
        sun.CreateIntensityAttr(6000.0)
        sun.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.85))
        xf = UsdGeom.Xformable(sun.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 30.0, 0.0))
    if not stage.GetPrimAtPath("/World/_Sky_").IsValid():
        sky = UsdLux.DomeLight.Define(stage, "/World/_Sky_")
        sky.CreateIntensityAttr(1200.0)
        sky.CreateColorAttr(Gf.Vec3f(0.55, 0.75, 1.0))


def add_characters(stage):
    """
    Wrapper + Char 子节点架构：
      Wrapper Xform → 我们控制位置和朝向
        Char Xform  → Mixamo USD 引用，自身 XformOp 完全保留
    """
    if not stage.GetPrimAtPath(CHAR_ROOT).IsValid():
        stage.DefinePrim(CHAR_ROOT, "Xform")

    for cfg in NPC_CONFIGS:
        wrapper_path = f"{CHAR_ROOT}/{cfg['name']}"
        char_path    = f"{wrapper_path}/Char"
        if stage.GetPrimAtPath(wrapper_path).IsValid():
            continue

        # Wrapper：只有 translate + rotateY
        wrapper = stage.DefinePrim(wrapper_path, "Xform")
        x, y, z = cfg["start"]
        w_xform = UsdGeom.Xformable(wrapper)
        w_xform.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(z)))
        w_xform.AddRotateYOp().Set(0.0)

        # Char：Mixamo USD 引用，不碰任何 XformOp
        char_prim = stage.DefinePrim(char_path, "Xform")
        char_prim.GetReferences().AddReference(cfg["usd"])

        carb.log_info(f"[Mixamo] Added {cfg['name']} at ({x:.1f},{y:.1f},{z:.1f})")


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("[Mixamo] NPC 巡逻演示  (UsdSkel 动画 + XformOp 移动)")
    print("="*60 + "\n")

    # 检查 Mixamo USD 是否存在
    import os
    if not os.path.exists(MIXAMO_CHAR_USD.replace("/", "\\")):
        print(f"[错误] Mixamo USD 文件不存在：{MIXAMO_CHAR_USD}")
        print("请先按照 docs/npc.md 流程准备角色 USD 文件：")
        print("  1. 从 mixamo.com 下载 FBX")
        print("  2. Blender 加 Root bone")
        print("  3. isaaclab.bat -p scripts/tools/convert_fbx_to_usd.py --input <fbx> --output <usd>")
        simulation_app.close()
        return

    # 获取动画时长（用于循环设置）
    anim_start, anim_end, anim_fps = get_anim_duration(MIXAMO_CHAR_USD)
    print(f"[Mixamo] 动画范围：{anim_start:.0f} ~ {anim_end:.0f} 帧  "
          f"({(anim_end - anim_start) / anim_fps:.2f} 秒 @ {anim_fps:.0f}fps)")

    print("[Step 1] 加载场景...")
    stage = load_scene()

    print("[Step 2] 设置灯光...")
    setup_lighting(stage)

    print("[Step 3] 添加 Mixamo 角色...")
    add_characters(stage)

    # 等待引用解析
    for _ in range(60):
        simulation_app.update()

    print("[Step 4] 启动时间轴（UsdSkel 动画随时间轴自动播放）...")
    # Stage 级别设置帧率
    stage.SetTimeCodesPerSecond(anim_fps)
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_start_time(anim_start / anim_fps)   # 秒为单位
    timeline.set_end_time(anim_end   / anim_fps)
    # 循环：部分 Isaac Sim 版本支持 set_looping，不支持则手动重置
    try:
        timeline.set_looping(True)
    except AttributeError:
        pass   # 不支持时在仿真循环里手动重置时间（见下方）
    timeline.play()
    for _ in range(10):
        simulation_app.update()

    agents = [PatrolAgent(cfg) for cfg in NPC_CONFIGS]

    print("[Step 5] 开始巡逻  (Ctrl+C 停止)")
    for cfg in NPC_CONFIGS:
        print(f"  {cfg['name']}: {cfg['start']} ↔ {cfg['end']}  {cfg['speed']} m/s")
    print("-"*60)

    frame        = 0
    dt           = 1.0 / FPS
    report_every = int(FPS * 2)
    anim_end_sec = anim_end / anim_fps

    try:
        while simulation_app.is_running():
            simulation_app.update()
            frame += 1

            # 动画循环兜底：若 set_looping 不支持，手动重置到起始时间
            if timeline.get_current_time() >= anim_end_sec:
                timeline.set_current_time(anim_start / anim_fps)

            for agent in agents:
                agent.step(stage, dt)

            if frame % report_every == 0:
                print(f"\n[Frame {frame}]  t={frame/FPS:.1f}s")
                for agent in agents:
                    p   = agent.pos
                    tgt = "→终点" if agent.going_forward else "←起点"
                    print(f"  {agent.name:12s}: ({p[0]:6.2f},{p[1]:4.2f},{p[2]:5.2f}) "
                          f"{agent.speed:.1f}m/s  {tgt}")

    except KeyboardInterrupt:
        print("\n[Mixamo] 用户停止.")
    finally:
        timeline.stop()
        simulation_app.close()


if __name__ == "__main__":
    main()
