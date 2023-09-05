from setuptools import setup, Extension
from torch.utils import cpp_extension

def get_install_requires():
    install_requires = [
        'torch'
    ]

    return install_requires

setup(name='custom_gridsample',
    ext_modules=[cpp_extension.CUDAExtension('custom_gridsample',
                    ['grid_sample.cpp', 'grid_sample_kernel.cu'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    python_requires='>=3.6',
    install_requires = get_install_requires(),
    author="Liuchun Yuan",
    author_email="ylc0003@gmail.com",
    description="Custom grid sample",
    )



