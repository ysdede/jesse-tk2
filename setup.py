from setuptools import find_packages, setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="jesse-tk2",
    version='0.0.7',
    packages=find_packages(),
    install_requires=required,

    entry_points='''
        [console_scripts]
        jesse-tk2=jessetk2.__init__:cli
    ''',
    python_requires='>=3.7',
    include_package_data=True,
)
