"""
场景物体类型检查器
==================
加载仓库场景（可选 NPC），遍历所有 USD prim，提取人能识别的物体类型清单。

三层信息提取策略：
  1. UsdSemantics.LabelsAPI — 官方语义标签（最可靠）
  2. prim 路径名推断 — 从 /World/Warehouse/Shelf_02 等路径提取 "Shelf"
  3. USD 引用资产名 — 从引用的 .usd 文件名提取 "SM_CardBoxD_04"

运行：
    isaaclab.bat -p projects/stereo_voxel/scripts/inspect_scene_objects.py
    isaaclab.bat -p projects/stereo_voxel/scripts/inspect_scene_objects.py --with_npc
    isaaclab.bat -p projects/stereo_voxel/scripts/inspect_scene_objects.py --headless
"""

import argparse
import os
import re
import sys
from collections import defaultdict

parser = argparse.ArgumentParser(description="Inspect all objects in warehouse scene")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--with_npc", action="store_true", help="Also load NPC characters via IRA")
parser.add_argument("--num_characters", type=int, default=3, help="Number of NPC characters")
parser.add_argument("--output", type=str, default=None, help="Save results to file")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

import carb
import omni.usd
from pxr import Sdf, Usd, UsdGeom, UsdSemantics

# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------
LOCAL_ASSETS_PATH = "D:/code/IsaacLab/Assets/Isaac/5.1"
if not os.path.isdir(LOCAL_ASSETS_PATH):
    try:
        from isaacsim.storage.native import get_assets_root_path
        LOCAL_ASSETS_PATH = get_assets_root_path()
    except Exception:
        pass
if not LOCAL_ASSETS_PATH or not os.path.isdir(LOCAL_ASSETS_PATH):
    print("[Inspector] ERROR: Cannot find Isaac assets.", file=sys.stderr)
    simulation_app.close()
    sys.exit(1)

ASSETS_ROOT = LOCAL_ASSETS_PATH.replace("\\", "/")
SCENE_USD = f"{ASSETS_ROOT}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"

# ---------------------------------------------------------------------------
# Load scene (same pattern as test_stereo_pair.py)
# ---------------------------------------------------------------------------
if args.with_npc:
    from isaacsim.core.utils.extensions import enable_extension
    NPC_EXTENSIONS = [
        "omni.anim.timeline", "omni.anim.graph.bundle", "omni.anim.graph.core",
        "omni.anim.retarget.core", "omni.anim.navigation.core",
        "omni.anim.navigation.bundle", "omni.anim.people", "omni.kit.scripting",
    ]
    for ext in NPC_EXTENSIONS:
        enable_extension(ext)
        simulation_app.update()

    # Pre-load scene
    import omni.kit.window.file
    old_ignore = carb.settings.get_settings().get("/app/file/ignoreUnsavedStage")
    carb.settings.get_settings().set("/app/file/ignoreUnsavedStage", True)
    omni.kit.window.file.open_stage(SCENE_USD, omni.usd.UsdContextInitialLoadSet.LOAD_ALL)
    carb.settings.get_settings().set("/app/file/ignoreUnsavedStage", old_ignore or False)

    for _ in range(60):
        simulation_app.update()

    # IRA NPC setup (with timeout, non-blocking)
    import tempfile
    enable_extension("isaacsim.replicator.agent.core")
    simulation_app.update()
    simulation_app.update()

    from isaacsim.replicator.agent.core.simulation import SimulationManager

    CHARACTER_MODELS = [
        "Isaac/People/Characters/F_Business_02/F_Business_02.usd",
        "Isaac/People/Characters/male_adult_construction_05_new/male_adult_construction_05_new.usd",
        "Isaac/People/Characters/female_adult_police_01_new/female_adult_police_01_new.usd",
    ]
    available = [m for m in CHARACTER_MODELS if os.path.isfile(f"{ASSETS_ROOT}/{m}")]
    num_chars = min(args.num_characters, len(available))

    if num_chars > 0:
        # Generate command file
        lines = []
        for i in range(num_chars):
            name = "Character" if i == 0 else f"Character_0{i}" if i < 10 else f"Character_{i}"
            lines.append(f"{name} GoTo 4.0 {i*3.0:.1f} 0.0 0")
            lines.append(f"{name} GoTo -4.0 {i*3.0:.1f} 0.0 180")
        cmd_file = os.path.join(tempfile.gettempdir(), "inspect_npc_cmd.txt")
        with open(cmd_file, "w") as f:
            f.write("\n".join(lines))

        char_folder = f"{ASSETS_ROOT}/Isaac/People/Characters/"
        config = (
            "isaacsim.replicator.agent:\n"
            "  version: 0.7.0\n"
            "  global:\n"
            "    seed: 42\n"
            "    simulation_length: 900\n"
            "  scene:\n"
            f"    asset_path: {SCENE_USD}\n"
            "  character:\n"
            f"    asset_path: {char_folder}\n"
            f"    command_file: inspect_npc_cmd.txt\n"
            f"    num: {num_chars}\n"
            "  replicator:\n"
            "    writer: IRABasicWriter\n"
            "    parameters:\n"
            "      rgb: false\n"
        )
        config_file = os.path.join(tempfile.gettempdir(), "inspect_npc_config.yaml")
        with open(config_file, "w") as f:
            f.write(config)

        sim_manager = SimulationManager()
        settings = carb.settings.get_settings()
        settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh", False)
        settings.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", True)

        if sim_manager.load_config_file(config_file):
            setup_done = [False]
            sim_manager.register_set_up_simulation_done_callback(lambda e: setup_done.__setitem__(0, True))
            sim_manager.set_up_simulation_from_config_file()
            tick = 0
            while not setup_done[0] and tick < 3000:
                simulation_app.update()
                tick += 1
            print(f"[Inspector] NPC setup {'done' if setup_done[0] else 'timeout'} at tick {tick}")

    for _ in range(30):
        simulation_app.update()
else:
    # Simple sublayer load
    stage = omni.usd.get_context().get_stage()
    stage.GetRootLayer().subLayerPaths.append(SCENE_USD)
    for _ in range(30):
        simulation_app.update()

print("[Inspector] Scene loaded. Scanning all prims...\n")

# ===========================================================================
# 核心：遍历所有 prim，提取物体类型信息
# ===========================================================================
stage = omni.usd.get_context().get_stage()


def get_semantic_labels(prim):
    """从 prim 获取 UsdSemantics.LabelsAPI 标签。"""
    labels = {}
    for schema_name in prim.GetAppliedSchemas():
        if schema_name.startswith("SemanticsLabelsAPI:"):
            instance_name = schema_name.split(":", 1)[1]
            sem_api = UsdSemantics.LabelsAPI(prim, instance_name)
            labels_attr = sem_api.GetLabelsAttr()
            if labels_attr:
                vals = labels_attr.Get()
                if vals:
                    labels[instance_name] = list(vals)
    return labels


def get_reference_asset_name(prim):
    """从 prim 的 USD 引用中提取资产文件名。"""
    refs = prim.GetMetadata("references")
    if refs:
        for ref_list in [refs.prependedItems, refs.appendedItems]:
            if ref_list:
                for ref in ref_list:
                    if hasattr(ref, "assetPath") and ref.assetPath:
                        basename = os.path.splitext(os.path.basename(ref.assetPath))[0]
                        return basename
    # Also check composition arcs
    if prim.HasAuthoredReferences():
        refs_api = prim.GetReferences()
        # Try to get from prim stack
        for spec in prim.GetPrimStack():
            ref_list = spec.referenceList
            if ref_list:
                for ref in ref_list.prependedItems:
                    if ref.assetPath:
                        return os.path.splitext(os.path.basename(ref.assetPath))[0]
    return None


def clean_prim_name(name):
    """从 prim 名称中提取人类可读的物体类型。
    例如: SM_CardBoxD_04_1847 → CardBoxD
          Shelf_02 → Shelf
          WallPanel_A → WallPanel
    """
    # 去掉常见前缀
    name = re.sub(r"^(SM_|S_|T_|M_|SK_|BP_)", "", name)
    # 去掉尾部数字编号（_02, _1847 等）
    name = re.sub(r"(_\d+)+$", "", name)
    # 去掉尾部单字母变体（_A, _B 等）
    name = re.sub(r"_[A-Z]$", "", name)
    return name


def classify_prim_type(prim):
    """综合判断 prim 代表的物体类型。"""
    path = prim.GetPath().pathString
    name = prim.GetName()
    prim_type = prim.GetTypeName()

    # 1. 语义标签（最优先）
    sem_labels = get_semantic_labels(prim)
    if sem_labels:
        all_labels = [l for vals in sem_labels.values() for l in vals]
        if all_labels:
            return all_labels[0], "semantic_label"

    # 2. 引用资产名
    ref_name = get_reference_asset_name(prim)
    if ref_name:
        cleaned = clean_prim_name(ref_name)
        if cleaned and len(cleaned) > 1:
            return cleaned, "reference_asset"

    # 3. prim 名称推断
    cleaned = clean_prim_name(name)
    if cleaned and len(cleaned) > 1:
        return cleaned, "prim_name"

    return None, None


# ---------------------------------------------------------------------------
# 遍历并收集
# ---------------------------------------------------------------------------
# 只关注有几何意义的 prim 类型
GEOMETRY_TYPES = {"Mesh", "Cube", "Sphere", "Cylinder", "Cone", "Capsule", "BasisCurves"}
XFORM_TYPE = "Xform"

# 收集结果
object_types = defaultdict(lambda: {"count": 0, "source": "", "paths": [], "prim_type": ""})
npc_characters = []
all_prim_count = 0
mesh_count = 0

for prim in stage.Traverse():
    all_prim_count += 1
    prim_type = prim.GetTypeName()
    path = prim.GetPath().pathString

    # 收集 NPC 角色
    if "Character" in path and prim_type == "SkelRoot":
        npc_characters.append(path)
        obj_name = prim.GetName()
        object_types[f"NPC:{obj_name}"]["count"] += 1
        object_types[f"NPC:{obj_name}"]["source"] = "skel_root"
        object_types[f"NPC:{obj_name}"]["prim_type"] = "SkelRoot"
        object_types[f"NPC:{obj_name}"]["paths"].append(path)
        continue

    # 只处理 Xform（作为物体容器）和几何体
    if prim_type == XFORM_TYPE:
        # Xform 如果有引用就是一个"物体"
        obj_type, source = classify_prim_type(prim)
        if obj_type and source in ("semantic_label", "reference_asset"):
            object_types[obj_type]["count"] += 1
            object_types[obj_type]["source"] = source
            object_types[obj_type]["prim_type"] = prim_type
            if len(object_types[obj_type]["paths"]) < 3:  # 只记前 3 个路径
                object_types[obj_type]["paths"].append(path)

    elif prim_type in GEOMETRY_TYPES:
        mesh_count += 1
        # 几何体：向上找父 Xform 的物体类型
        parent = prim.GetParent()
        if parent and parent.GetTypeName() == XFORM_TYPE:
            obj_type, source = classify_prim_type(parent)
            if not obj_type:
                obj_type, source = classify_prim_type(prim)
        else:
            obj_type, source = classify_prim_type(prim)

        if obj_type:
            if obj_type not in object_types or object_types[obj_type]["count"] == 0:
                object_types[obj_type]["source"] = source
                object_types[obj_type]["prim_type"] = prim_type
            object_types[obj_type]["count"] += 1
            if len(object_types[obj_type]["paths"]) < 3:
                object_types[obj_type]["paths"].append(path)

    # 相机
    elif prim_type == "Camera":
        cam_name = prim.GetName()
        object_types[f"Camera:{cam_name}"]["count"] += 1
        object_types[f"Camera:{cam_name}"]["source"] = "prim_type"
        object_types[f"Camera:{cam_name}"]["prim_type"] = "Camera"
        object_types[f"Camera:{cam_name}"]["paths"].append(path)

    # 灯光
    elif "Light" in prim_type:
        light_name = prim_type  # DomeLight, DistantLight, RectLight, etc.
        object_types[f"Light:{light_name}"]["count"] += 1
        object_types[f"Light:{light_name}"]["source"] = "prim_type"
        object_types[f"Light:{light_name}"]["prim_type"] = prim_type
        if len(object_types[f"Light:{light_name}"]["paths"]) < 3:
            object_types[f"Light:{light_name}"]["paths"].append(path)


# ===========================================================================
# 输出结果
# ===========================================================================
print("=" * 80)
print(f"  场景物体类型清单 — {SCENE_USD.split('/')[-1]}")
print(f"  总 prim 数: {all_prim_count}, 几何体(Mesh等): {mesh_count}")
if npc_characters:
    print(f"  NPC 角色: {len(npc_characters)}")
print("=" * 80)

# 按来源和数量排序
sorted_types = sorted(object_types.items(), key=lambda x: (-x[1]["count"]))

# 分组输出
categories = {
    "NPC / 角色": [],
    "灯光": [],
    "相机": [],
    "场景物体": [],
}

for name, info in sorted_types:
    if name.startswith("NPC:"):
        categories["NPC / 角色"].append((name, info))
    elif name.startswith("Light:"):
        categories["灯光"].append((name, info))
    elif name.startswith("Camera:"):
        categories["相机"].append((name, info))
    else:
        categories["场景物体"].append((name, info))

output_lines = []

for cat_name, items in categories.items():
    if not items:
        continue
    header = f"\n{'─'*40}\n  {cat_name} ({len(items)} 种)\n{'─'*40}"
    print(header)
    output_lines.append(header)

    for name, info in items:
        line = f"  {name:<40s}  x{info['count']:<5d}  [{info['source']}]"
        print(line)
        output_lines.append(line)
        # 示例路径
        for p in info["paths"][:2]:
            detail = f"      └─ {p}"
            print(detail)
            output_lines.append(detail)

# 去重类型列表
print(f"\n{'─'*40}")
print(f"  去重物体类型列表 (共 {len(object_types)} 种)")
print(f"{'─'*40}")

unique_list = sorted(object_types.keys())
for i, name in enumerate(unique_list):
    line = f"  {i+1:3d}. {name}"
    print(line)
    output_lines.append(line)

# 保存到文件
if args.output:
    out_path = args.output
else:
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "scene_object_types.txt")

with open(out_path, "w", encoding="utf-8") as f:
    f.write(f"场景: {SCENE_USD}\n")
    f.write(f"总 prim: {all_prim_count}, 几何体: {mesh_count}, NPC: {len(npc_characters)}\n")
    f.write(f"去重物体种类: {len(object_types)}\n\n")
    for line in output_lines:
        f.write(line + "\n")
    f.write(f"\n--- 去重类型列表 ---\n")
    for name in unique_list:
        info = object_types[name]
        f.write(f"{name}\t{info['count']}\t{info['source']}\n")

print(f"\n[Inspector] 结果已保存: {out_path}")

# GUI 模式下保持窗口打开
if not args.headless:
    print("[Inspector] GUI 模式，可在 Stage 面板查看场景结构。关闭窗口退出。")
    while simulation_app.is_running():
        simulation_app.update()

simulation_app.close()
