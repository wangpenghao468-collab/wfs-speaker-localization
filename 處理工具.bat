@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=%USERPROFILE%\anaconda3\envs\speaker-mask\python.exe"
if not exist "%PY%" set "PY=python"
echo 啟動重建處理工具（只在這台電腦上，自動挑沒人用的連接埠）...
"%PY%" tools\webtool\server.py
pause
