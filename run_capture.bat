@echo off
cd /d D:\code\IsaacLab
echo [%date% %time%] Starting capture > run_log.txt

REM Kill any leftover kit processes
taskkill /F /IM kit.exe >nul 2>&1
timeout /t 5 /nobreak >nul

call _isaac_sim\python.bat projects\stereo_voxel\scripts\stereo_voxel_capture.py --headless --num_frames 5 --capture_interval 30 --no_npc >> run_log.txt 2>&1
echo [%date% %time%] Exit code: %ERRORLEVEL% >> run_log.txt
