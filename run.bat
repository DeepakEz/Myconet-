@echo off
REM Contemplative MycoNet Quick Run Script for Windows

echo === Contemplative MycoNet Quick Run ===
echo.

if "%1"=="" goto default
if "%1"=="minimal" goto minimal
if "%1"=="min" goto minimal
if "%1"=="basic" goto basic
if "%1"=="advanced" goto advanced
if "%1"=="adv" goto advanced
if "%1"=="list" goto list

echo Unknown experiment: %1
echo Use 'run.bat list' to see available experiments
goto end

:minimal
echo Running Minimal Test...
python myconet_contemplative_main.py --config-file "configs/minimal_test.json" --verbose --seed 42
goto end

:basic
echo Running Basic Contemplative Experiment...
python myconet_contemplative_main.py --config-file "configs/basic_test.json" --verbose --seed 42
goto end

:advanced
echo Running Advanced Contemplative Study...
python myconet_contemplative_main.py --config-file "configs/advanced_study.json" --verbose --seed 42
goto end

:list
echo Available experiments:
echo   minimal  - Quick test (5 agents, 100 steps)
echo   basic    - Basic experiment (10 agents, 300 steps)
echo   advanced - Full study (20 agents, 1000 steps)
echo.
echo Usage: run.bat [experiment_name]
echo Example: run.bat minimal
goto end

:default
echo No experiment specified. Running minimal test...
echo.
python myconet_contemplative_main.py --config-file "configs/minimal_test.json" --verbose --seed 42

:end
echo.
pause
