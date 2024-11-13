from distutils.core import setup, Extension
import numpy

module = Extension('fslic',
                   sources=['fslic_wrapper.c', 'fslic.c'],
                   include_dirs=[numpy.get_include()],
                   extra_compile_args=['-std=c99'])

setup(name='fslic',
      version='1.0',
      description='FSLIC algorithm implementation',
      ext_modules=[module])