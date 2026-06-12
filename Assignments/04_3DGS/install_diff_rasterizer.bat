@echo off
chcp 65001 >nul
set VSLANG=1033
set PYTHONUTF8=1
set TORCH_DONT_CHECK_COMPILER_ABI=1
set DISTUTILS_USE_SDK=1
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set TORCH_CUDA_ARCH_LIST=12.0
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
python -m pip install --no-build-isolation gaussian-splatting-main\submodules\diff-gaussian-rasterization
