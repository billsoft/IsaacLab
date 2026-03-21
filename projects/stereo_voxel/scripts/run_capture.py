"""Launcher: runs stereo_voxel_capture.py via Isaac Sim Python with output logging.

Usage: python projects/stereo_voxel/scripts/run_capture.py [extra args...]
  Default: --headless --num_frames 5 --capture_interval 30 --no_npc
"""
import subprocess
import sys
import os
import time

ISAAC_PYTHON = r"D:\code\IsaacLab\_isaac_sim\python.bat"
SCRIPT = r"D:\code\IsaacLab\projects\stereo_voxel\scripts\stereo_voxel_capture.py"
LOG = r"D:\code\IsaacLab\run_log.txt"

# Kill any leftover kit processes
os.system("taskkill /F /IM kit.exe >nul 2>&1")
time.sleep(3)

# Default args, can be overridden via command line
if len(sys.argv) > 1:
    args = sys.argv[1:]
else:
    args = ["--headless", "--num_frames", "5", "--capture_interval", "30", "--no_npc"]

with open(LOG, "w") as log:
    log.write(f"CMD: {ISAAC_PYTHON} {SCRIPT} {' '.join(args)}\n")
    log.flush()
    proc = subprocess.Popen(
        [ISAAC_PYTHON, "-u", SCRIPT] + args,
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=r"D:\code\IsaacLab",
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    proc.wait()
    log.write(f"\nEXIT CODE: {proc.returncode}\n")

print(f"Done. Exit code: {proc.returncode}. Log: {LOG}")
