from distutils.core import setup, Extension

import os
os.environ['CC'] = 'g++'
os.environ['CXX'] = 'g++'
os.environ['CPP'] = 'g++'
#os.environ['LDSHARED'] = 'g++'

setup(name='numerical',
      ext_modules=[Extension('numerical', 
                             ['DiscreteUniformGeneratorMt19937_wrap.cpp'],
                             language='c++')])

#extra_link_args=['-undefined dynamic_lookup', '-bundle']
#, '-undefined dynamic_lookup'

#setup(name='numerical',
#      ext_modules=[Extension('numerical', 
#                             ['DiscreteUniformGeneratorMt19937.i'],
#                             language='c++',
#                             swig_opts=['-c++'])],
#      options={'build_ext':{'swig_opts':'-c++'}})
