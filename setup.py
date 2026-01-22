import os
import sys
import sysconfig
import subprocess
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str):
        super().__init__(name, sources=[])
        self.sourcedir = str(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except Exception as e:
            raise RuntimeError("CMake is required to build this project.") from e
        super().run()

    def build_extension(self, ext: CMakeExtension):
        ext_fullpath = Path(self.get_ext_fullpath(ext.name)).resolve()
        extdir = ext_fullpath.parent
        extdir.mkdir(parents=True, exist_ok=True)

        cfg = "Debug" if self.debug else "Release"
        build_temp = Path(self.build_temp) / ext.name.replace(".", "_")
        build_temp.mkdir(parents=True, exist_ok=True)

        python_exe = sys.executable

        # ✅ 检测 HIP 环境（关键）
        is_hip = "ROCM_PATH" in os.environ or "/opt/rocm" in os.environ.get("PATH", "")
        
        # ✅ 修改 2：改用 ROCm 架构环境变量
        # gpu_arch = os.environ.get("FASTEQ_ROCM_ARCH", "").strip()
        gpu_arch = os.environ.get("FASTEQ_ROCM_ARCH", "gfx90a").strip()
        
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DPython3_EXECUTABLE={python_exe}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DGPU_ARCHS={gpu_arch}",  # ✅ 传递给 CMake
        ]

        # if is_hip:
        #     cmake_args.append("-DWITH_HIP=ON")  # ✅ 通知 CMake 这是 HIP 构建

        if is_hip:
            cmake_args.append("-DWITH_HIP=ON")
            # 合并伪造路径和 PyTorch 路径（用分号分隔）
            import torch
            torch_cmake_path = os.path.join(os.path.dirname(torch.__file__), "share", "cmake")
            cmake_args.append(f"-DCMAKE_PREFIX_PATH=/home/wangsen/.local/lib/cmake;{torch_cmake_path}")

        if not os.environ.get("CMAKE_GENERATOR"):
            cmake_args += ["-GNinja"]

        # ✅ 并行编译任务数
        build_args = ["--config", cfg]
        jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "10")
        if hasattr(self, "parallel") and self.parallel:
            jobs = str(self.parallel)
        build_args += ["-j", jobs]

        # 配置
        subprocess.check_call(
            ["cmake", ext.sourcedir, *cmake_args],
            cwd=str(build_temp),
        )
        
        # 构建
        subprocess.check_call(
            ["cmake", "--build", ".", *build_args],
            cwd=str(build_temp),
        )

        # ✅ 查找生成的库（优先 _hip.so）
        candidates = list(extdir.glob("_hip.so"))
        if not candidates:
            candidates = list(extdir.glob("_cuda.so"))
        
        if not candidates:
            raise RuntimeError(f"找不到编译生成的库文件 in {extdir}")

        built = candidates[0]
        if built.resolve() != ext_fullpath.resolve():
            self.copy_file(str(built), str(ext_fullpath))


ROOT = Path(__file__).resolve().parent
CMAKE_SOURCE_DIR = ROOT / "fasteq" / "hip"  # ✅ HIP 源目录

setup(
    name="fasteq",
    version="0.1.0",
    description="FastEq HIP extensions for AMD GPUs",  # ✅ 更新描述
    python_requires=">=3.10",
    packages=find_packages(where="."),
    include_package_data=True,
    ext_modules=[
        CMakeExtension("fasteq.hip._hip", sourcedir=str(CMAKE_SOURCE_DIR)),  # ✅ HIP 扩展名
    ],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)

