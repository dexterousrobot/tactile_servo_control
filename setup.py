import os
from setuptools import setup

lib_folder = os.path.dirname(os.path.realpath(__file__))

# get required packages from requirements.txt
requirement_path = os.path.join(lib_folder, 'requirements.txt')
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

setup(name='tactile_servo_control',
      version='0.0.1',
      description='Implementation of tactile pose prediction and servo control.',
      author='Alex Church, Nathan Lepora, Yijiong Lin',
      author_email='alex.church@bristol.ac.uk, n.lepora@bristol.ac.uk,  yijionglin.bourne@gmail.com,',
      license='',
      packages=['tactile_servo_control'],
      install_requires=install_requires,
      zip_safe=False)
