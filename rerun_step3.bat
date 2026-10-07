@echo off
setlocal EnableExtensions
rem ============================================================
rem  rerun_step3.bat - Run the full Matrix-3D pipeline (low-VRAM)
rem
rem  Runs all 3 steps with --low-vram, so Step 2 uses the 5B
rem  video model (~12 GB VRAM) and the whole pipeline fits on a
rem  24 GB GPU (e.g. RTX 3090).  Total ~1-3 h depending on GPU.
rem
rem    Step 1: text/image -^> panorama image
rem    Step 2: panorama   -^> video  (5B model, --low-vram)
rem    Step 3: video      -^> 3D scene (3DGS training, 3000 iters)
rem
rem  NOTE: needs ~14 GB of FREE GPU memory. If llama.cpp (or
rem  anything else) is holding the GPU, stop it first - the
rem  script checks and refuses to start otherwise.
rem
rem  Usage:  double-click, or from a terminal:  rerun_step3.bat
rem  Force start without the GPU check:  set SKIP_GPU_CHECK=1
rem ============================================================

set "PROJ=%~dp0"
if "%PROJ:~-1%"=="\" set "PROJ=%PROJ:~0,-1%"

rem --- Convert the Windows project path to its WSL /mnt/... form ---
rem (WSL mount points are lowercase: /mnt/d, not /mnt/D)
set "WSL_PROJ="
for /f "tokens=* delims=" %%P in ('powershell -NoProfile -Command "$p='%PROJ%'; '/mnt/' + $p.Substring(0,1).ToLower() + $p.Substring(2).Replace('\','/')"') do set "WSL_PROJ=%%P"

set "MIN_FREE_MB=14000"

echo Project dir : %PROJ%
echo WSL path    : %WSL_PROJ%

rem --- Timestamp for the log file ---
set "TS="
for /f "tokens=* delims=" %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%T"
set "LOG=%PROJ%\pipeline_lowvram_%TS%.log"

rem --- GPU free-memory check (skip with SKIP_GPU_CHECK=1) ---
if /i "%SKIP_GPU_CHECK%"=="1" goto run
set "FREE_MB="
for /f "skip=1 tokens=1" %%M in ('nvidia-smi --query-gpu=memory.free --format=csv') do set "FREE_MB=%%M"
if not defined FREE_MB (
    echo ERROR: nvidia-smi failed - is the NVIDIA driver running?
    pause
    exit /b 1
)
echo Free GPU memory: %FREE_MB% MB (need ^>= %MIN_FREE_MB% MB)
if %FREE_MB% LSS %MIN_FREE_MB% (
    echo.
    echo ERROR: not enough free GPU memory.
    echo        Another process is holding the GPU, e.g. llama.cpp.
    echo        Stop it and re-run this script, or set SKIP_GPU_CHECK=1 to force.
    pause
    exit /b 1
)

:run
echo Log file    : %LOG%
echo Starting full pipeline with --low-vram (Step 1 -^> 2 -^> 3, ~1-3 h)...
echo.

wsl -- bash -c "cd %WSL_PROJ% && bash wsl_generate.sh --low-vram 2>&1 | tee pipeline_lowvram_%TS%.log"
set "RC=%ERRORLEVEL%"

echo.
if exist "%PROJ%\output\example1\generated_3dgs_opt.ply" (
    echo ============================================================
    echo SUCCESS: output\example1\generated_3dgs_opt.ply was saved.
    echo ============================================================
) else (
    echo ============================================================
    echo FAILURE: generated_3dgs_opt.ply is missing, exit code %RC%.
    echo Check the log: %LOG%
    echo ============================================================
)
pause
endlocal
