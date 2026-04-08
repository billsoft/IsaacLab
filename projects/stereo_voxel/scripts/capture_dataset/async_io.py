"""异步 IO 模块
================
ThreadPoolExecutor 后台保存图像/DNG/体素/JSON，避免阻塞渲染线程。
"""

import json
import os
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import numpy as np


_save_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="saver_v2")
_pending_saves: list[Future] = []


def async_save_image(path: str, rgb_array: np.ndarray):
    """异步保存 RGB numpy array 为 PNG。"""
    def _save(p, arr):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(p, bgr)
    future = _save_pool.submit(_save, path, rgb_array.copy())
    _pending_saves.append(future)


def async_save_dng(path: str, rgb_array: np.ndarray, raw_converter, dng_writer):
    """异步保存 RGB → 12-bit Bayer RAW → DNG。"""
    def _save(p, arr, conv, writer):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        raw = conv.rgb_to_raw(arr.astype(np.uint8))
        writer.write(raw, p)
    future = _save_pool.submit(_save, path, rgb_array.copy(), raw_converter, dng_writer)
    _pending_saves.append(future)


def async_save_voxel(path_prefix: str, semantic: np.ndarray,
                     instance: np.ndarray,
                     flow: np.ndarray | None = None,
                     flow_mask: np.ndarray | None = None,
                     orientation: np.ndarray | None = None,
                     angular_vel: np.ndarray | None = None):
    """异步保存体素数据（semantic + instance + flow + orientation + angular_vel）。"""
    def _save(pp, sem, ins, fl, fm, ori, av):
        os.makedirs(os.path.dirname(pp), exist_ok=True)
        np.savez_compressed(f"{pp}_semantic.npz", data=sem)
        np.savez_compressed(f"{pp}_instance.npz", data=ins.astype(np.uint16))
        if fl is not None and fm is not None:
            kw = {"flow": fl, "flow_mask": fm}
            if ori is not None:
                kw["orientation"] = ori
            if av is not None:
                kw["angular_vel"] = av
            np.savez_compressed(f"{pp}_flow.npz", **kw)
    future = _save_pool.submit(
        _save, path_prefix,
        semantic.copy(), instance.copy(),
        flow.copy() if flow is not None else None,
        flow_mask.copy() if flow_mask is not None else None,
        orientation.copy() if orientation is not None else None,
        angular_vel.copy() if angular_vel is not None else None,
    )
    _pending_saves.append(future)


def async_save_json(path: str, data: dict):
    """异步保存 JSON。"""
    def _save(p, d):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(d, f, indent=2)
    future = _save_pool.submit(_save, path, data)
    _pending_saves.append(future)


def wait_pending_saves():
    """等待所有异步保存完成。"""
    for f in _pending_saves:
        f.result()
    _pending_saves.clear()


def shutdown():
    """关闭线程池。"""
    wait_pending_saves()
    _save_pool.shutdown(wait=True)
