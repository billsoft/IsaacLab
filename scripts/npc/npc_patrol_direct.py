"""
NPC 直线巡逻演示 v3 —— Wrapper 架构，不干扰角色自身变换
============================================================
核心修复：
  之前：直接在角色 prim 上 ClearXformOpOrder() → 删除了角色 USD 自带的
        站立修正旋转 → 角色侧身倒地
  现在：角色 USD 挂在 Char 子节点，Wrapper 只控制位置和朝向：

    /World/Characters/
      Worker_01/           ← Wrapper Xform（translate + rotateY，我们每帧更新）
        Char/              ← 角色 USD 引用（完全不碰它的 XformOp）

运行：
    isaaclab.bat -p scripts/npc/npc_patrol_direct.py
    isaaclab.bat -p scripts/npc/npc_patrol_direct.py --headless
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="NPC patrol demo v3 - wrapper arch")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── AppLauncher 之后再 import ─────────────────────────────────────────────
import time
import numpy as np
import carb
import omni.kit.app
import omni.usd
import omni.timeline
from pxr import Gf, UsdGeom, UsdLux

# ── 路径常量 ──────────────────────────────────────────────────────────────
ASSET_ROOT  = "D:/code/IsaacLab/Assets/Isaac/5.1"
SCENE_USD   = f"{ASSET_ROOT}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
PEOPLE_ROOT = f"{ASSET_ROOT}/Isaac/People/Characters"
CHAR_ROOT   = "/World/Characters"

# ── NPC 配置 ──────────────────────────────────────────────────────────────
NPC_CONFIGS = [
    {
        "name":  "Worker_01",
        "usd":   f"{PEOPLE_ROOT}/male_adult_construction_01_new/male_adult_construction_01_new.usd",
        "start": np.array([ 9.0, 0.0,  2.0]),
        "end":   np.array([-9.0, 0.0,  2.0]),
        "speed": 1.2,
    },
    {
        "name":  "Worker_02",
        "usd":   f"{PEOPLE_ROOT}/F_Business_02/F_Business_02.usd",
        "start": np.array([ 9.0, 0.0,  0.0]),
        "end":   np.array([-9.0, 0.0,  0.0]),
        "speed": 1.0,
    },
    {
        "name":  "Worker_03",
        "usd":   f"{PEOPLE_ROOT}/F_Medical_01/F_Medical_01.usd",
        "start": np.array([ 9.0, 0.0, -2.0]),
        "end":   np.array([-9.0, 0.0, -2.0]),
        "speed": 1.4,
    },
]

FPS = 60.0


# ─────────────────────────────────────────────────────────────────────────────
# 巡逻控制器（只操作 Wrapper Xform，不触及角色自身 prim）
# ─────────────────────────────────────────────────────────────────────────────
class PatrolAgent:
    """每帧更新 Wrapper prim 的 translate 和 rotateY"""

    def __init__(self, cfg: dict):
        self.name  = cfg["name"]
        self.start = cfg["start"].copy()
        self.end   = cfg["end"].copy()
        self.speed = cfg["speed"]
        self.pos   = cfg["start"].copy().astype(float)
        self.going_forward = True

        # Wrapper prim 路径（不是角色 prim，而是我们自己定义的 Xform）
        self.wrapper_path = f"{CHAR_ROOT}/{self.name}"

        # 朝向角：arctan2(dx, dz) 适用于 Isaac Sim 默认 +Z 朝向的角色
        d = self.end - self.start
        self._yaw_fwd = float(np.degrees(np.arctan2(d[0], d[2])))
        self._yaw_bwd = (self._yaw_fwd + 180.0) % 360.0

        self._translate_op = None
        self._rotate_op    = None

    def _init_ops(self, stage):
        """
        从 Wrapper prim 取出（或创建）translate / rotateY op。
        注意：Wrapper 是我们自己创建的纯 Xform，不是角色 prim，
        所以这里操作 Wrapper 的 XformOp 是安全的。
        """
        if self._translate_op is not None:
            return True
        prim = stage.GetPrimAtPath(self.wrapper_path)
        if not prim.IsValid():
            return False
        xform = UsdGeom.Xformable(prim)
        # 取出 add_characters 已经建好的 op
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                self._translate_op = op
            elif op.GetOpType() == UsdGeom.XformOp.TypeRotateY:
                self._rotate_op = op
        if self._translate_op is None:
            # 兜底：重建（正常情况不会走这里）
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
        self.pos = target.copy() if step_dist >= dist else self.pos + direction / dist * step_dist

        # 写入 Wrapper XformOp
        self._translate_op.Set(Gf.Vec3d(float(self.pos[0]),
                                         float(self.pos[1]),
                                         float(self.pos[2])))
        yaw = self._yaw_fwd if self.going_forward else self._yaw_bwd
        if self._rotate_op:
            self._rotate_op.Set(float(yaw))

        return self.pos.copy()


# ─────────────────────────────────────────────────────────────────────────────
# 场景加载
# ─────────────────────────────────────────────────────────────────────────────
def load_scene():
    ctx = omni.usd.get_context()
    carb.log_info(f"[NPC] Opening {SCENE_USD}")
    ctx.open_stage(SCENE_USD)
    for _ in range(400):
        simulation_app.update()
        if ctx.get_stage().GetPrimAtPath("/World").IsValid():
            break
        time.sleep(0.05)
    carb.log_info("[NPC] Scene loaded.")
    return ctx.get_stage()


def setup_lighting(stage):
    sun_path = "/World/_Sun_"
    if not stage.GetPrimAtPath(sun_path).IsValid():
        sun = UsdLux.DistantLight.Define(stage, sun_path)
        sun.CreateIntensityAttr(6000.0)
        sun.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.85))
        xf = UsdGeom.Xformable(sun.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 30.0, 0.0))

    sky_path = "/World/_Sky_"
    if not stage.GetPrimAtPath(sky_path).IsValid():
        sky = UsdLux.DomeLight.Define(stage, sky_path)
        sky.CreateIntensityAttr(1200.0)
        sky.CreateColorAttr(Gf.Vec3f(0.55, 0.75, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# 角色添加：Wrapper + Char 子节点，不清除角色自身 XformOp
# ─────────────────────────────────────────────────────────────────────────────
def add_characters(stage):
    """
    /World/Characters/
      Worker_01/           ← Wrapper Xform（translate + rotateY）
        Char/              ← 角色 USD 引用（自身 XformOp 原样保留）
    """
    if not stage.GetPrimAtPath(CHAR_ROOT).IsValid():
        stage.DefinePrim(CHAR_ROOT, "Xform")

    for cfg in NPC_CONFIGS:
        wrapper_path = f"{CHAR_ROOT}/{cfg['name']}"
        char_path    = f"{wrapper_path}/Char"

        if stage.GetPrimAtPath(wrapper_path).IsValid():
            continue

        # 1. Wrapper Xform：只负责世界坐标定位和朝向
        wrapper = stage.DefinePrim(wrapper_path, "Xform")
        x, y, z = cfg["start"]
        w_xform = UsdGeom.Xformable(wrapper)
        w_xform.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(z)))
        w_xform.AddRotateYOp().Set(0.0)   # 朝向后续由 PatrolAgent 每帧更新

        # 2. Char 子节点：只添加 USD 引用，完全不碰 XformOp
        char_prim = stage.DefinePrim(char_path, "Xform")
        char_prim.GetReferences().AddReference(cfg["usd"])
        # ↑ 不调用 ClearXformOpOrder，让角色自带的站立/旋转修正变换完整保留

        carb.log_info(f"[NPC] Added {cfg['name']} wrapper={wrapper_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 启动后检查并打印角色的实际初始方向（帮助诊断朝向问题）
# ─────────────────────────────────────────────────────────────────────────────
def inspect_char_ops(stage):
    """打印每个角色 Char 子节点上的 XformOp，用于确认角色自带变换"""
    for cfg in NPC_CONFIGS:
        char_path = f"{CHAR_ROOT}/{cfg['name']}/Char"
        prim = stage.GetPrimAtPath(char_path)
        if not prim.IsValid():
            print(f"  [inspect] {char_path} not found")
            continue
        xform = UsdGeom.Xformable(prim)
        ops = xform.GetOrderedXformOps()
        desc = ", ".join(str(op.GetOpType()) for op in ops) if ops else "(none)"
        print(f"  [inspect] {cfg['name']}/Char  ops: {desc}")


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("[NPC] 巡逻演示 v3  (Wrapper 架构)")
    print("="*60 + "\n")

    print("[Step 1] 加载场景...")
    stage = load_scene()

    print("[Step 2] 设置灯光...")
    setup_lighting(stage)

    print("[Step 3] 添加角色（Wrapper + Char 架构）...")
    add_characters(stage)

    # 等待 USD 引用解析完成
    for _ in range(60):
        simulation_app.update()

    print("[Step 3b] 角色自带 XformOp 检查：")
    inspect_char_ops(stage)

    print("[Step 4] 启动仿真时间轴...")
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_start_time(0)
    timeline.set_end_time(1e9)
    timeline.play()
    for _ in range(10):
        simulation_app.update()

    agents = [PatrolAgent(cfg) for cfg in NPC_CONFIGS]

    print("[Step 5] 开始巡逻仿真  (Ctrl+C 停止)")
    print("  说明：角色侧身倒地 = 角色自带变换已被修正（不清除），应已站立")
    print("  说明：T-pose 为无动画驱动的默认姿态，位置移动仍正常")
    for cfg in NPC_CONFIGS:
        print(f"  {cfg['name']}: {cfg['start']} ↔ {cfg['end']}  speed={cfg['speed']} m/s")
    print("-"*60)

    frame        = 0
    dt           = 1.0 / FPS
    report_every = int(FPS * 2)

    try:
        while simulation_app.is_running():
            simulation_app.update()
            frame += 1

            for agent in agents:
                agent.step(stage, dt)

            if frame % report_every == 0:
                print(f"\n[NPC] Frame {frame}  (t={frame/FPS:.1f}s)")
                for agent in agents:
                    p   = agent.pos
                    tgt = "→终点" if agent.going_forward else "←起点"
                    print(f"  {agent.name:12s}: ({p[0]:6.2f}, {p[1]:4.2f}, {p[2]:5.2f})"
                          f"  {agent.speed:.1f} m/s  {tgt}")

    except KeyboardInterrupt:
        print("\n[NPC] 用户停止.")
    finally:
        timeline.stop()
        simulation_app.close()


if __name__ == "__main__":
    main()
