from setuptools import setup, find_packages

setup(
    name = 'ImgProcess',
    version = '0.0.1',
    author = 'Fly',
    packages = find_packages(),
    install_requires = ['numpy', 'tensorflow', 'matplotlib', 'opencv-python', 'Pillow'],
    package_data = {'ImgProcess': ['model/*']},
    include_package_data = True,
)