from distutils.core import setup, Extension

module = Extension('mcqd',
                   sources = ['mcqd_api.cpp'])

setup (name = 'mcqd',
       description = 'Package for finding a maximal clique',
       ext_modules = [module])
