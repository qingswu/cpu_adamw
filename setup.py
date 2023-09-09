import os
from setuptools import find_packages, setup
import torch
from torch.utils import cpp_extension

if __name__ == "__main__":
    sources = [
        "cpu_adamw/csrc/cpu_adam_impl.cpp",
        "cpu_adamw/csrc/cpu_adam.cpp",
    ]

    include_dirs = [f"{os.path.dirname(os.path.abspath(__file__))}/cpu_adamw/csrc/"]
    if torch.cuda.is_available():
        extra_compile_args = {
            "cxx": [
                "-g",
                "-L/usr/lib/x86_64-linux-gnu",
                "-O2",
                "-std=c++17",
                "-D__ENABLE_CUDA__",
            ],
            "nvcc": ["-O2"],
        }
        module = cpp_extension.CUDAExtension(
            name="_cpu_adamw",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        )
    else:
        extra_compile_args = {
            "cxx": ["-g", "-L/usr/lib/x86_64-linux-gnu", "-O2", "-std=c++17"]
        }
        module = cpp_extension.CppExtension(
            name="_cpu_adamw",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        )
    setup(
        name="cpu-adamw",
        version="0.0.1",
        description="CPU AdamW optimizer based on DeepSpeed",
        packages=find_packages(),
        install_requires=["pybind11", "py-cpuinfo"],
        ext_modules=[module],
        cmdclass={"build_ext": cpp_extension.BuildExtension},
    )
