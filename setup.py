from distutils.core import setup
from setuptools import find_packages

setup(
    name='azkaban',
    packages=find_packages(exclude=['test', '*.test', '*.test.*']),
    version='0.1.1',
    description='RL multi-agent communicative and collaborative environment',
    author='Nikita Lapkov <nikita.lapkov@gmail.com>, justheuristic <jheuristic@yandex-team.ru>',
    url='https://github.com/laplab/azkaban',
    keywords=['rl', 'agent', 'environment', 'communication', 'collaboration'],
)
