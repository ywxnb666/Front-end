@echo off
setlocal

cd /d "%~dp0"

pyinstaller ^
  --noconfirm ^
  --clean ^
  --onefile ^
  --windowed ^
  --name RemoteCloneTool ^
  --collect-all paramiko ^
  main.py

endlocal
