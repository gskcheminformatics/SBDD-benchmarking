from setuptools import setup, find_packages

setup(name='sbdd_bench',
   version='0.0.1',
   description='Benchmark for structure-based molecule generation methods',
   url='https://github.com/gsk-tech/SBDD-benchmarking',
   author = 'Natasha Sanjrani',
   packages=find_packages(),
   package_data={'sbdd_bench': ['sbdd_analysis/dpocket_crystal_info.txt']}
)
