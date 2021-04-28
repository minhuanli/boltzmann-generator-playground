#!/usr/bin/env python

from setuptools import setup, find_packages

packages=find_packages() 

setup(name='Deep Boltzmann Package',
      version='0.1',
      description='Deep Learning Methods for Sampling Boltzmann Distributions',
      author='Frank Noe',
      author_email='frank.noe@fu-berlin.de',
      url='',
      #packages=['deep_boltzmann'],
      packages=packages
     )