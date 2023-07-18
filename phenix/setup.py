import sys, platform
import setuptools

with open("README.md", "r") as file:
    long_desc = file.read()

setuptools.setup(
    name='ArchPy',
    version='0.1.0',
    author="Ludovic Schorpp",
    author_email="ludovic.schorpp@unine.ch",
    description="Simpler geological and property models",
    long_description=long_desc,
    long_description_content_type='text/markdown',
    url = "https://github.com/randlab/ArchPy",
    install_requires=['matplotlib', 'numpy', 'pyvista', 'scipy'],
    packages=setuptools.find_packages(),
    include_package_data=True,
    data_files=[("lib\\site-packages\\ArchPy\\libraries", ["ArchPy\\libraries\\cov_facies.dll"])],
    license=open('LICENSE', encoding='utf-8').read()
)