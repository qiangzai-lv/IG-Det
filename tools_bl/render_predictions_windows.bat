@echo off
setlocal

rem Set BLENDER_EXE to use a specific Blender installation.
if defined BLENDER_EXE goto found_blender

for /f "delims=" %%B in ('where blender.exe 2^>nul') do (
    set "BLENDER_EXE=%%B"
    goto found_blender
)

for /d %%D in ("%ProgramFiles%\Blender Foundation\Blender *") do (
    if exist "%%~D\blender.exe" set "BLENDER_EXE=%%~D\blender.exe"
)

:found_blender
if not defined BLENDER_EXE (
    echo Blender was not found. Set BLENDER_EXE to the path of blender.exe.
    exit /b 1
)
if not exist "%BLENDER_EXE%" (
    echo Blender executable not found: "%BLENDER_EXE%"
    exit /b 1
)

"%BLENDER_EXE%" -b --python-exit-code 1 --python "%~dp0render_predictions.py" -- %*
exit /b %errorlevel%
