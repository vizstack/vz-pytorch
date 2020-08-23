from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='vz-pytorch',
    version='0.1.0',
    license='MIT',
    packages=['vz_pytorch'],
    zip_safe=False,
    author='Ryan Holmdahl & Nikhil Bhattasali',
    author_email='vizstack@gmail.com',
    install_requires=['vizstack-py', 'torch'],
    description="Create Vizstack visualizations of Pytorch models.",
    url='https://github.com/vizstack/vz-pytorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
)
