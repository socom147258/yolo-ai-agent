@echo off
chcp 65001 >nul
echo ========================================
echo   YOLO 缺陷检测系统 - 启动中...
echo ========================================
echo.

cd /d "%~dp0"

REM 优先使用 .venv，如果不存在则尝试 venv
if exist ".venv\Scripts\python.exe" (
    echo [信息] 使用虚拟环境: .venv
    set "PYTHON=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    echo [信息] 使用虚拟环境: venv
    set "PYTHON=venv\Scripts\python.exe"
) else (
    echo [警告] 未找到虚拟环境，尝试使用系统 Python
    set "PYTHON=python"
)

echo.
echo [启动] Streamlit 服务...
echo [提示] 浏览器会自动打开，关闭此窗口即可停止服务
echo.

start "" "%PYTHON%" -m streamlit run app.py --server.port 8501

timeout /t 3 /nobreak >nul
echo.
echo ========================================
echo   服务已启动！
echo   本地地址: http://localhost:8501
echo ========================================
echo.
pause
