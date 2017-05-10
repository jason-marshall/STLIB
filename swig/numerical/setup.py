from distutils.core import setup, Extension

import os
os.environ['CC'] = 'g++'
os.environ['CXX'] = 'g++'
os.environ['CPP'] = 'g++'

setup(name='numerical',
      ext_modules=[Extension('numerical', 
                             ['numerical_wrap.cpp'],
                             language='c++')])
