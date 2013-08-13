#!/usr/bin/env python

from distutils.core import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs
import os

include_dirs = [os.getcwd()]

module = Extension('_mcqd',
                   include_dirs = include_dirs +  get_numpy_include_dirs(),
                   sources = ['_mcqd.cpp'],
                   language = 'c++')

setup (name = 'mcqd',
       description = 'Package for finding a maximal clique',
       ext_modules = [module])
