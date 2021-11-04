#!/usr/bin/env python

from setuptools import setup, find_packages

packages=find_packages() 

setup(name='deep_boltzmann',
      version='0.2',
      description='Deep Learning Methods for Sampling Boltzmann Distributions, Originally By Frank Noe, Updates to support TF 2.0 by Minhuan Li',
      author='Frank Noe, Minhuan Li',
      author_email='frank.noe@fu-berlin.de, minhuanli@g.harvard.edu',
      url='',
      #packages=['deep_boltzmann'],
      packages=packages
     )