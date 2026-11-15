@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================
echo [1/5] 构建镜像 zhixun:latest
echo ============================================
docker build --platform=linux/amd64 -t zhixun:latest .
if errorlevel 1 (
  echo [FAIL] 构建失败，终止
  exit /b 1
)

echo ============================================
echo [2/5] 镜像内 Python 版本（应为 3.10.x）
echo ============================================
docker run --rm --platform=linux/amd64 zhixun:latest python --version
if errorlevel 1 (
  echo [FAIL] 镜像内 python 不可用
  exit /b 1
)

echo ============================================
echo [3/5] 镜像内依赖清单
echo ============================================
docker run --rm zhixun:latest pip list
if errorlevel 1 (
  echo [FAIL] pip list 失败
  exit /b 1
)

echo ============================================
echo [4/5] 复现自检 reproduce.py
echo ============================================
docker run --rm -v "%~dp0outputs:/app/outputs" zhixun:latest python reproduce.py
if errorlevel 1 (
  echo [FAIL] 复现自检失败
  exit /b 1
)

echo ============================================
echo [5/5] 正式闭环 main.py（离线时 SwanLab 自动走本地队列，不影响退出码）
echo ============================================
docker run --rm -v "%~dp0outputs:/app/outputs" zhixun:latest python main.py
if errorlevel 1 (
  echo [FAIL] main.py 退出码非 0
  exit /b 1
)

echo ============================================
echo 全部 5 步通过，镜像可提交
echo ============================================
endlocal
