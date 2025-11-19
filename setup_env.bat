@echo off
echo === Setting up MuJoCo Environment ===

REM Create virtual environment
python -m venv .venv

REM Activate venv
call .\.venv\Scripts\activate

REM Upgrade pip
pip install --upgrade pip

REM Install MuJoCo and viewer
pip install mujoco mujoco-python-viewer numpy opencv-python

echo === Environment ready! ===
echo Run: ".venv\Scripts\activate" to activate the environment next time.