@echo off
REM LabelMe SAM2 Propagate 启动脚本
REM 使用方法：labelme_sam2_tool.bat <图像文件夹路径>

REM ============ 配置区（修改成你的实际路径）============
set SAM2_PYTHON=C:\Users\Admin1\miniforge3\envs\sam2\python.exe
set SCRIPT_PATH=%~dp0sam2_propagate.py
REM ====================================================

if "%~1"=="" (
    echo 错误：未提供图像文件夹路径
    echo 使用方法：%~nx0 ^<图像文件夹路径^>
    pause
    exit /b 1
)

echo ========================================
echo LabelMe SAM2 视频传播工具
echo ========================================
echo.
echo 目标文件夹: %~1
echo Python环境: %SAM2_PYTHON%
echo 脚本路径: %SCRIPT_PATH%
echo.
echo 开始处理...
echo ----------------------------------------
echo.

"%SAM2_PYTHON%" "%SCRIPT_PATH%" --dir "%~1" --preview

echo.
echo ----------------------------------------
echo 处理完成！
echo 预览图已保存到：%~1\_preview\
echo.
pause
