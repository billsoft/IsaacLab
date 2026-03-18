"""
语义体素占用网格 (Semantic Occupancy Grid) 分类定义
=====================================================

基于 SemanticKITTI / Occ3D 体素占用预测任务的 18 类语义标签体系，
适配室内仓库双目鱼眼俯视场景。

体素状态三态模型：
  - free (class 0)        : 确认为空（空气），该空间无物体
  - occupied (class 1-16)  : 确认被占用，且可分类到具体语义
  - general (class 17)     : 确认被占用，但不属于 1-16 中任何一类
  - unobserved             : 未被任何传感器观测到（训练/评估时忽略）

注意：unobserved 不是一个语义类别，而是一个标注状态。
      在体素 ground truth 中用特殊值（如 255）表示。

评估指标：
  - mIoU (mean Intersection-over-Union) — 主要指标，对 class 1-17 计算
  - 每类 IoU — 各类别交并比
  - class 0 (free) 单独评估几何完成度 (Scene Completion IoU)
"""

# ============================================================================
# 类别 ID 定义
# ============================================================================

# 空闲 / 空气（1 类）
FREE = 0

# 交通参与者（8 类）—— 动态物体
CAR = 1
BICYCLE = 2
MOTORCYCLE = 3
TRUCK = 4
OTHER_VEHICLE = 5
PERSON = 6          # 行人（NPC / 真人）
BICYCLIST = 7       # 骑自行车的人（人+车整体）
MOTORCYCLIST = 8    # 骑摩托车的人（人+车整体）

# 地面 / 可行驶区域（4 类）—— 静态平面
ROAD = 9            # 道路（仓库中：主通道）
PARKING = 10        # 停车区（仓库中：货物暂存区）
SIDEWALK = 11       # 人行道（仓库中：人行通道标线区域）
OTHER_GROUND = 12   # 其他地面（仓库中：未标记地面）

# 环境结构（4 类）—— 静态结构体
BUILDING = 13       # 建筑/人造结构（仓库墙壁、立柱、货架框架）
FENCE = 14          # 围栏/栏杆（仓库中：安全围栏、隔断）
VEGETATION = 15     # 植被（仓库中：少见，但室外场景有）
TRUNK = 16          # 树干/柱状物（仓库中：立柱、管道）

# 通用占位（1 类）
GENERAL_OBJECT = 17  # 有占位但不属于上述 16 类（箱子、托盘、工具等）

# 特殊标注值（不参与语义分类）
UNOBSERVED = 255     # 未观测区域（训练/评估忽略）


# ============================================================================
# 类别名称映射
# ============================================================================

CLASS_NAMES = {
    0:  "free",
    1:  "car",
    2:  "bicycle",
    3:  "motorcycle",
    4:  "truck",
    5:  "other-vehicle",
    6:  "person",
    7:  "bicyclist",
    8:  "motorcyclist",
    9:  "road",
    10: "parking",
    11: "sidewalk",
    12: "other-ground",
    13: "building",
    14: "fence",
    15: "vegetation",
    16: "trunk",
    17: "general-object",
}

# 中文名（用于日志和可视化标注）
CLASS_NAMES_ZH = {
    0:  "空闲/空气",
    1:  "汽车",
    2:  "自行车",
    3:  "摩托车",
    4:  "卡车",
    5:  "其他车辆",
    6:  "行人",
    7:  "骑行者(自行车)",
    8:  "骑行者(摩托车)",
    9:  "道路/主通道",
    10: "停车区/暂存区",
    11: "人行道/标线通道",
    12: "其他地面",
    13: "建筑/墙壁/货架",
    14: "围栏/隔断",
    15: "植被",
    16: "树干/立柱",
    17: "通用物体",
}


# ============================================================================
# 可视化颜色 (RGB, 0-255)
# ============================================================================
# 配色参考 SemanticKITTI 官方 + 仓库场景可辨识度优化

CLASS_COLORS = {
    0:  (0,   0,   0),      # free — 黑色（通常不渲染）
    1:  (100, 150, 245),    # car — 浅蓝
    2:  (100, 230, 245),    # bicycle — 青色
    3:  (30,  60,  150),    # motorcycle — 深蓝
    4:  (80,  30,  180),    # truck — 紫色
    5:  (255, 30,  30),     # other-vehicle — 红色
    6:  (255, 40,  200),    # person — 粉色 ★ 仓库主要动态目标
    7:  (150, 30,  90),     # bicyclist — 暗粉
    8:  (255, 0,   255),    # motorcyclist — 品红
    9:  (255, 150, 255),    # road — 浅粉
    10: (75,  0,   75),     # parking — 暗紫
    11: (175, 0,   75),     # sidewalk — 玫红
    12: (255, 200, 0),      # other-ground — 金黄
    13: (255, 120, 50),     # building — 橙色 ★ 仓库主要静态结构
    14: (255, 240, 150),    # fence — 浅黄
    15: (0,   175, 0),      # vegetation — 绿色
    16: (135, 60,  0),      # trunk — 棕色
    17: (150, 240, 80),     # general-object — 黄绿 ★ 箱子/托盘等
}


# ============================================================================
# 类别分组（用于分组评估和分析）
# ============================================================================

# 动态物体（可能帧间移动）
DYNAMIC_CLASSES = {CAR, BICYCLE, MOTORCYCLE, TRUCK, OTHER_VEHICLE,
                   PERSON, BICYCLIST, MOTORCYCLIST}

# 静态地面
GROUND_CLASSES = {ROAD, PARKING, SIDEWALK, OTHER_GROUND}

# 静态结构
STRUCTURE_CLASSES = {BUILDING, FENCE, VEGETATION, TRUNK}

# 所有占用类别（排除 free）
OCCUPIED_CLASSES = set(range(1, 18))

# 仓库场景高频类别（评估重点关注）
WAREHOUSE_PRIMARY = {PERSON, BUILDING, GENERAL_OBJECT, ROAD, OTHER_GROUND}

NUM_CLASSES = 18  # 0-17 共 18 类（含 free）
NUM_SEMANTIC = 17  # 1-17 共 17 个语义类（不含 free）


# ============================================================================
# 仓库场景适配说明
# ============================================================================
#
# 本项目场景为 Isaac Sim Simple_Warehouse，俯视双目鱼眼相机 + NPC 行人。
# 与户外自动驾驶场景的差异：
#
# | 类别         | 户外场景           | 仓库场景                    |
# |-------------|--------------------|-----------------------------|
# | car/truck   | 常见               | 少见（叉车/AGV 归入此类）     |
# | person      | 中频               | 高频（NPC 行人是主要目标）    |
# | road        | 柏油路面           | 仓库主通道地面               |
# | parking     | 停车位             | 货物暂存区                   |
# | sidewalk    | 人行道             | 人行标线通道                 |
# | building    | 建筑外墙           | 墙壁、立柱、货架金属框架     |
# | fence       | 路边围栏           | 安全围栏、区域隔断           |
# | vegetation  | 常见               | 极少（室内几乎没有）         |
# | trunk       | 树干               | 立柱、管道                   |
# | general-obj | 交通锥/垃圾桶等    | 箱子、托盘、工具、货物 ★     |
#
# 仓库场景中 general-object (17) 的占比会远高于户外场景，
# 因为大量货物/箱子/托盘不属于 1-16 中的任何标准类别。


# ============================================================================
# Isaac Sim 场景物体 → 语义类别映射表
# ============================================================================
#
# 由 inspect_scene_objects.py 扫描 Simple_Warehouse 场景得到 80 种物体类型，
# 逐一映射到 0-17 语义类别。供体素标注、训练、数据集生成程序调用。
#
# 映射规则：
#   - 精确匹配优先（完整物体名）
#   - 前缀匹配兜底（如 "Wall" 开头的都归 BUILDING）
#   - Camera/Light prim 不占体素空间，映射到 FREE(0)
#   - 未匹配的可见几何体默认归 GENERAL_OBJECT(17)
#
# 使用方式：
#   from semantic_classes import lookup_class_id
#   class_id = lookup_class_id("CardBoxA")  # → 17 (GENERAL_OBJECT)
#   class_id = lookup_class_id("forklift")  # → 4  (TRUCK)
# ============================================================================

# ── 精确映射表：Isaac Sim 物体名 → 语义类别 ID ──
# 键为 inspect_scene_objects.py 输出的去重物体类型名（clean 后）
OBJECT_TO_CLASS = {
    # ─────────────────────────────────────────────────────────────
    # FREE (0) — 非物理实体，不占体素
    # ─────────────────────────────────────────────────────────────
    # Camera / Light prim 是渲染辅助，不产生几何占用
    # 由前缀规则处理，不在此表中逐一列出

    # ─────────────────────────────────────────────────────────────
    # TRUCK (4) — 大型工程车辆
    # ─────────────────────────────────────────────────────────────
    "forklift":             TRUCK,          # 叉车 → 大型工程车辆
    "ForkliftFork":         TRUCK,          # 叉车货叉部分

    # ─────────────────────────────────────────────────────────────
    # OTHER_VEHICLE (5) — 其他小型车辆/手推车
    # ─────────────────────────────────────────────────────────────
    "PushcartA":            OTHER_VEHICLE,  # 手推车 → 小型搬运车辆

    # ─────────────────────────────────────────────────────────────
    # PERSON (6) — 行人 / NPC 角色
    # ─────────────────────────────────────────────────────────────
    "Biped_Setup":              PERSON,     # NPC 动画骨骼系统
    "Body_Mesh":                PERSON,     # NPC 身体网格
    "biped_demo_meters":        PERSON,     # NPC 演示骨骼
    "female_adult_business":    PERSON,     # 女性商务 NPC
    "female_adult_business_02": PERSON,     # 女性商务 NPC 变体
    "female_adult_police_01":   PERSON,     # 女性警察 NPC
    "female_adult_police_03":   PERSON,     # 女性警察 NPC 变体
    "male_adult_construction":  PERSON,     # 男性工人 NPC
    "male_adult_construction_01": PERSON,   # 男性工人 NPC 变体
    "male_adult_construction_02": PERSON,   # 男性工人 NPC 变体
    "male_adult_construction_03": PERSON,   # 男性工人 NPC 变体
    "male_adult_construction_05": PERSON,   # 男性工人 NPC 变体
    "M_Medical_01":             PERSON,     # 男性医疗 NPC
    "F_Business_02":            PERSON,     # 女性商务 NPC

    # ─────────────────────────────────────────────────────────────
    # ROAD (9) — 主通道地面
    # ─────────────────────────────────────────────────────────────
    "floor02":              ROAD,           # 仓库主地面 → 主通道

    # ─────────────────────────────────────────────────────────────
    # PARKING (10) — 停放区 / 暂存区
    # ─────────────────────────────────────────────────────────────
    "FloorDecal_Keepclear":     PARKING,    # "保持通畅"地贴 → 暂存区标记
    "FloorDecal_QuadRed2x1":   PARKING,    # 红色方形地贴 → 停放标记区

    # ─────────────────────────────────────────────────────────────
    # SIDEWALK (11) — 人行标线通道
    # ─────────────────────────────────────────────────────────────
    "FloorDecal_RecRed1X1":       SIDEWALK, # 红色矩形地贴 → 人行标线
    "FloorDecal_StripeFull_4m":   SIDEWALK, # 条纹地贴 → 人行安全通道标线

    # ─────────────────────────────────────────────────────────────
    # OTHER_GROUND (12) — 其他地面
    # ─────────────────────────────────────────────────────────────
    "GroundPlane":          OTHER_GROUND,   # 基础地平面

    # ─────────────────────────────────────────────────────────────
    # BUILDING (13) — 建筑结构 / 墙壁 / 天花板 / 货架框架
    # ─────────────────────────────────────────────────────────────
    "WallA_3M":             BUILDING,       # 墙壁 A 型 3m
    "WallA_6M":             BUILDING,       # 墙壁 A 型 6m
    "WallA_InnerCorner":    BUILDING,       # 墙壁 A 型内角
    "WallB_3M":             BUILDING,       # 墙壁 B 型 3m
    "WallB_6M":             BUILDING,       # 墙壁 B 型 6m
    "WallB_InnerCorner":    BUILDING,       # 墙壁 B 型内角
    "CeilingA_6X6":         BUILDING,       # 天花板
    "BeamA_9M":             BUILDING,       # 横梁 9m
    "BracketBeam":          BUILDING,       # 支架横梁
    "BracketBeam_3m":       BUILDING,       # 支架横梁 3m
    "BracketSlot":          BUILDING,       # 支架插槽
    "RackFrame":            BUILDING,       # 货架金属框架
    "RackShelf":            BUILDING,       # 货架层板
    "LampCeilingA":         BUILDING,       # 天花板灯具（固定在建筑上）

    # ─────────────────────────────────────────────────────────────
    # FENCE (14) — 围栏 / 护栏 / 标识牌 / 安全标识
    # ─────────────────────────────────────────────────────────────
    "WallWire":             FENCE,          # 铁丝网墙 → 围栏
    "Rackshield":           FENCE,          # 货架护栏
    "AisleSign":            FENCE,          # 过道标识牌
    "SignA":                FENCE,          # 标识牌 A
    "SignB":                FENCE,          # 标识牌 B
    "SignCVer":             FENCE,          # 标识牌 C（竖版）
    "EmergencyBoardFull":   FENCE,          # 应急信息板
    "WetFloorSign":         FENCE,          # 小心地滑标识
    "TrafficCone":          FENCE,          # 交通锥 → 路障/隔离设施

    # ─────────────────────────────────────────────────────────────
    # TRUNK (16) — 立柱 / 柱状结构
    # ─────────────────────────────────────────────────────────────
    "PillarA_9M":           TRUNK,          # 立柱 9m
    "PillarPartA_9M":       TRUNK,          # 立柱部件 9m

    # ─────────────────────────────────────────────────────────────
    # GENERAL_OBJECT (17) — 通用占位物体（箱子/瓶子/桶/托盘等）
    # ─────────────────────────────────────────────────────────────
    "CardBoxA":             GENERAL_OBJECT, # 纸箱 A
    "CardBoxB":             GENERAL_OBJECT, # 纸箱 B
    "CardBoxC":             GENERAL_OBJECT, # 纸箱 C
    "CardBoxD":             GENERAL_OBJECT, # 纸箱 D
    "BottlePlasticA":       GENERAL_OBJECT, # 塑料瓶 A
    "BottlePlasticB":       GENERAL_OBJECT, # 塑料瓶 B
    "BottlePlasticC":       GENERAL_OBJECT, # 塑料瓶 C
    "BottlePlasticD":       GENERAL_OBJECT, # 塑料瓶 D
    "BottlePlasticE":       GENERAL_OBJECT, # 塑料瓶 E
    "BarelPlastic":         GENERAL_OBJECT, # 塑料桶
    "BucketPlastic":        GENERAL_OBJECT, # 塑料桶（小）
    "CratePlastic":         GENERAL_OBJECT, # 塑料箱
    "CratePlasticNote":     GENERAL_OBJECT, # 塑料箱（带标签）
    "PaletteA":             GENERAL_OBJECT, # 木托盘
    "RackPile":             GENERAL_OBJECT, # 货架堆叠物
    "FireExtinguisher":     GENERAL_OBJECT, # 灭火器
    "FuseBox":              GENERAL_OBJECT, # 配电箱
    "Barcode":              GENERAL_OBJECT, # 条形码标签
    "PaperNote_Small":      GENERAL_OBJECT, # 纸条/便签
    "Paper_Shortcut":       GENERAL_OBJECT, # 纸张
}


# ── 前缀匹配规则（兜底，按优先级从高到低排列）──
# 当精确匹配 OBJECT_TO_CLASS 未命中时，按前缀逐一尝试
OBJECT_PREFIX_RULES = [
    # (前缀,              类别ID,          说明)
    ("NPC:",              PERSON,          "NPC 角色"),
    ("Camera:",           FREE,            "相机 prim（不占体素）"),
    ("Light:",            FREE,            "灯光 prim（不占体素）"),
    ("Wall",              BUILDING,        "各类墙壁"),
    ("Ceiling",           BUILDING,        "天花板"),
    ("Beam",              BUILDING,        "横梁"),
    ("Bracket",           BUILDING,        "支架"),
    ("Rack",              BUILDING,        "货架结构"),
    ("Pillar",            TRUNK,           "立柱"),
    ("Floor",             ROAD,            "地面"),
    ("FloorDecal",        SIDEWALK,        "地面标线"),
    ("Sign",              FENCE,           "标识牌"),
    ("CardBox",           GENERAL_OBJECT,  "纸箱"),
    ("Bottle",            GENERAL_OBJECT,  "瓶子"),
    ("Crate",             GENERAL_OBJECT,  "塑料箱"),
    ("Barrel",            GENERAL_OBJECT,  "桶"),
    ("Paper",             GENERAL_OBJECT,  "纸张"),
    ("female_adult",      PERSON,          "女性 NPC"),
    ("male_adult",        PERSON,          "男性 NPC"),
    ("biped",             PERSON,          "NPC 骨骼"),
    ("Forklift",          TRUCK,           "叉车"),
    ("forklift",          TRUCK,           "叉车"),
    ("Pushcart",          OTHER_VEHICLE,   "手推车"),
]


def lookup_class_id(object_type_name: str) -> int:
    """根据 Isaac Sim 物体类型名查找对应的语义类别 ID。

    查找顺序：
      1. OBJECT_TO_CLASS 精确匹配
      2. OBJECT_PREFIX_RULES 前缀匹配
      3. 默认返回 GENERAL_OBJECT (17)

    Args:
        object_type_name: inspect_scene_objects.py 输出的物体类型名

    Returns:
        int: 语义类别 ID (0-17)

    Examples:
        >>> lookup_class_id("forklift")
        4
        >>> lookup_class_id("CardBoxA")
        17
        >>> lookup_class_id("NPC:female_adult_business_02")
        6
        >>> lookup_class_id("Camera:OmniverseKit_Persp")
        0
        >>> lookup_class_id("SomeUnknownObject")
        17
    """
    # 1. 精确匹配
    if object_type_name in OBJECT_TO_CLASS:
        return OBJECT_TO_CLASS[object_type_name]

    # 2. 前缀匹配
    for prefix, class_id, _ in OBJECT_PREFIX_RULES:
        if object_type_name.startswith(prefix):
            return class_id

    # 3. 默认：通用占位
    return GENERAL_OBJECT


def lookup_class_name(object_type_name: str) -> str:
    """根据 Isaac Sim 物体类型名返回语义类别英文名。"""
    return CLASS_NAMES[lookup_class_id(object_type_name)]


def lookup_class_name_zh(object_type_name: str) -> str:
    """根据 Isaac Sim 物体类型名返回语义类别中文名。"""
    return CLASS_NAMES_ZH[lookup_class_id(object_type_name)]


def get_mapping_summary() -> dict:
    """返回所有已注册映射的统计摘要，按语义类别分组。

    Returns:
        dict: {class_id: {"name": str, "name_zh": str, "objects": list[str]}}
    """
    summary = {}
    for cid in range(NUM_CLASSES):
        summary[cid] = {
            "name": CLASS_NAMES[cid],
            "name_zh": CLASS_NAMES_ZH[cid],
            "objects": [],
        }

    # 从精确映射收集
    for obj_name, cid in OBJECT_TO_CLASS.items():
        summary[cid]["objects"].append(obj_name)

    return summary


# ============================================================================
# 自测：打印映射覆盖情况
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  语义类别 ← Isaac Sim 物体映射表")
    print("=" * 70)

    summary = get_mapping_summary()
    for cid in range(NUM_CLASSES):
        info = summary[cid]
        objs = info["objects"]
        color = CLASS_COLORS[cid]
        print(f"\n  [{cid:2d}] {info['name']:<20s} {info['name_zh']:<15s}  "
              f"RGB{color}  ({len(objs)} 种物体)")
        for obj in objs:
            print(f"        - {obj}")

    # 验证 80 种物体的覆盖率
    scene_objects = [
        "AisleSign", "Barcode", "BarelPlastic", "BeamA_9M", "Biped_Setup",
        "Body_Mesh", "BottlePlasticA", "BottlePlasticB", "BottlePlasticC",
        "BottlePlasticD", "BottlePlasticE", "BracketBeam", "BracketBeam_3m",
        "BracketSlot", "BucketPlastic", "Camera:Camera", "Camera:Camera_01",
        "Camera:OmniverseKit_Persp", "Camera:OmniverseKit_Front",
        "CardBoxA", "CardBoxB", "CardBoxC", "CardBoxD", "CeilingA_6X6",
        "CratePlastic", "CratePlasticNote", "EmergencyBoardFull",
        "FireExtinguisher", "FloorDecal_Keepclear", "FloorDecal_QuadRed2x1",
        "FloorDecal_RecRed1X1", "FloorDecal_StripeFull_4m", "ForkliftFork",
        "FuseBox", "GroundPlane", "LampCeilingA", "Light:DistantLight",
        "Light:RectLight", "NPC:biped_demo_meters",
        "NPC:female_adult_business_02", "NPC:male_adult_construction_01",
        "NPC:male_adult_construction_02", "PaletteA", "PaperNote_Small",
        "Paper_Shortcut", "PillarA_9M", "PillarPartA_9M", "PushcartA",
        "RackFrame", "RackPile", "RackShelf", "Rackshield", "SignA",
        "SignB", "SignCVer", "TrafficCone", "WallA_3M", "WallA_6M",
        "WallA_InnerCorner", "WallB_3M", "WallB_6M", "WallB_InnerCorner",
        "WallWire", "WetFloorSign", "female_adult_business", "floor02",
        "forklift", "male_adult_construction",
    ]

    print(f"\n{'='*70}")
    print(f"  覆盖率验证: {len(scene_objects)} 种场景物体")
    print(f"{'='*70}")

    by_class = {}
    for obj in scene_objects:
        cid = lookup_class_id(obj)
        cname = CLASS_NAMES[cid]
        by_class.setdefault(cid, []).append(obj)

    for cid in sorted(by_class.keys()):
        objs = by_class[cid]
        print(f"\n  [{cid:2d}] {CLASS_NAMES[cid]:<20s} ({len(objs)} 种)")
        for obj in objs:
            print(f"        {obj}")

    uncovered = [o for o in scene_objects if lookup_class_id(o) == GENERAL_OBJECT
                 and o not in OBJECT_TO_CLASS]
    if uncovered:
        print(f"\n  ⚠ 以下物体未精确匹配，由前缀/默认规则归入 GENERAL_OBJECT:")
        for o in uncovered:
            print(f"        {o}")
    else:
        print(f"\n  ✓ 所有 {len(scene_objects)} 种物体均已覆盖")
