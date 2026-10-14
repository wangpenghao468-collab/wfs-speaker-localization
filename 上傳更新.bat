@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=%USERPROFILE%\anaconda3\envs\speaker-mask\python.exe"
if not exist "%PY%" set "PY=python"
echo [1/3] build site
"%PY%" tools\build_site.py || goto :fail
echo [2/3] commit
git add -A
git diff --cached --quiet && (echo no changes & goto :done)
git commit -m "update %date% %time:~0,5%" || goto :fail
echo [3/3] push
git push || goto :fail
echo done: https://wangpenghao468-collab.github.io/wfs-speaker-localization/ (1-2 min)
:done
pause
exit /b 0
:fail
echo FAILED - see the message above
pause
exit /b 1
