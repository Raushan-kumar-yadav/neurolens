@echo off
rem Loads the Visual Studio compiler, then configures and builds neurolens.
rem   build.cmd                          build everything
rem   build.cmd --target neurolens_xor   build one target
rem Always use this (or F7 / a Developer PowerShell) instead of a bare "cmake --preset dev":
rem a plain shell cannot find cl, and the failed configure poisons build\CMakeCache.txt.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cd /d "%~dp0"

rem Heal a cache poisoned by a failed configure (it picks msys64 tools instead of MSVC ones).
if exist build\CMakeCache.txt (
    findstr /i /c:"msys64" build\CMakeCache.txt >nul && (
        echo build.cmd: stale build cache found, recreating the build folder
        rmdir /s /q build
    )
)

cmake --preset dev || exit /b 1
cmake --build --preset dev %*
