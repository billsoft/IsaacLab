"""实例 ID 管理器
================
按语义类别分段分配实例 ID，用于区分动态/静态物体。

ID 范围约定（参见 方向预测.md §3.2）：
  0          : 背景（所有静态物体合并）
  1~999      : 车辆
  1000~1999  : 行人
  2000~2999  : 两轮车
  3000~9999  : 其他动态
  65535      : 未观测
"""

from __future__ import annotations

import json
import sys
import os

# 添加 scripts 目录到 path，复用现有 semantic_classes
_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from semantic_classes import (
    CAR, TRUCK, OTHER_VEHICLE,
    PERSON, BICYCLIST, MOTORCYCLIST,
    BICYCLE, MOTORCYCLE,
    DYNAMIC_CLASSES,
)


# 语义类 → 实例类别
_CLASS_TO_CATEGORY = {
    CAR: "vehicle",
    TRUCK: "vehicle",
    OTHER_VEHICLE: "vehicle",
    PERSON: "pedestrian",
    BICYCLIST: "pedestrian",
    MOTORCYCLIST: "pedestrian",
    BICYCLE: "cyclist",
    MOTORCYCLE: "cyclist",
}


class InstanceRegistry:
    """分段实例 ID 分配器。

    同一个 prim_path 在整个采集过程中保持同一 ID（跨帧一致）。
    """

    ID_RANGES: dict[str, tuple[int, int]] = {
        "vehicle":    (1, 999),
        "pedestrian": (1000, 1999),
        "cyclist":    (2000, 2999),
        "other":      (3000, 9999),
    }
    BACKGROUND: int = 0
    UNOBSERVED: int = 65535

    def __init__(self):
        self._prim_to_id: dict[str, int] = {}
        self._id_to_info: dict[int, dict] = {}
        self._next_ids: dict[str, int] = {
            cat: lo for cat, (lo, _) in self.ID_RANGES.items()
        }
        self._frame_first_seen: dict[int, str] = {}
        self._frame_last_seen: dict[int, str] = {}

    @staticmethod
    def category_for_class(class_id: int) -> str | None:
        """语义类 ID → 实例类别名。静态类返回 None。"""
        return _CLASS_TO_CATEGORY.get(class_id)

    @staticmethod
    def is_dynamic_class(class_id: int) -> bool:
        return class_id in DYNAMIC_CLASSES

    def get_or_assign(self, prim_path: str, semantic_class_id: int,
                      frame_id: str = "") -> int:
        """获取或分配实例 ID。

        静态物体返回 BACKGROUND (0)。
        动态物体按类别从对应范围分配唯一 ID。
        """
        if prim_path in self._prim_to_id:
            iid = self._prim_to_id[prim_path]
            if frame_id:
                self._frame_last_seen[iid] = frame_id
            return iid

        category = self.category_for_class(semantic_class_id)
        if category is None:
            return self.BACKGROUND

        lo, hi = self.ID_RANGES[category]
        new_id = self._next_ids[category]
        if new_id > hi:
            print(f"[instance] WARNING: {category} ID range exhausted "
                  f"({lo}~{hi}), wrapping")
            new_id = lo

        self._prim_to_id[prim_path] = new_id
        self._id_to_info[new_id] = {
            "prim_path": prim_path,
            "semantic_class": semantic_class_id,
            "category": category,
        }
        self._next_ids[category] = new_id + 1

        if frame_id:
            self._frame_first_seen[new_id] = frame_id
            self._frame_last_seen[new_id] = frame_id

        return new_id

    def is_dynamic(self, instance_id: int) -> bool:
        """实例 ID 是否对应动态物体。"""
        return instance_id != self.BACKGROUND and instance_id != self.UNOBSERVED

    def get_all_dynamic_ids(self) -> list[int]:
        return [iid for iid in self._id_to_info if self.is_dynamic(iid)]

    def to_json(self) -> dict:
        """输出 instance_meta.json 内容。"""
        instances = {}
        for iid, info in sorted(self._id_to_info.items()):
            entry = {
                "prim_path": info["prim_path"],
                "semantic_class": info["semantic_class"],
                "category": info["category"],
            }
            if iid in self._frame_first_seen:
                entry["first_frame"] = self._frame_first_seen[iid]
            if iid in self._frame_last_seen:
                entry["last_frame"] = self._frame_last_seen[iid]
            instances[str(iid)] = entry

        return {
            "version": "2.0",
            "total_instances": len(self._id_to_info),
            "id_ranges": {
                "background": self.BACKGROUND,
                "vehicle": list(self.ID_RANGES["vehicle"]),
                "pedestrian": list(self.ID_RANGES["pedestrian"]),
                "cyclist": list(self.ID_RANGES["cyclist"]),
                "other_dynamic": list(self.ID_RANGES["other"]),
                "unobserved": self.UNOBSERVED,
            },
            "instances": instances,
        }

    def save(self, path: str):
        """保存 instance_meta.json。"""
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)
        print(f"[instance] Saved {len(self._id_to_info)} instances → {path}")
