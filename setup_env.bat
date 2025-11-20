@echo off
echo === Setting up MuJoCo Environment ===

REM Create virtual environment
python -m venv .venv

REM Activate venv
call .\.venv\Scripts\activate

REM Upgrade pip
pip install --upgrade pip

REM Install MuJoCo and viewer
pip install mujoco==3.2.0 mujoco-python-viewer numpy opencv-python

echo === Environment ready! ===
echo Run: ".venv\Scripts\activate" to activate the environment next time.
echo Use "deactivate" to exit the virtual environment.

echo For Unix-like systems, use the following command to activate:
echo source .venv/bin/activate to activate on Unix-like systems.
echo Use "deactivate" to exit the virtual environment on Unix-like systems.
echo === End of setup ===