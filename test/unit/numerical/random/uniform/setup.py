from distutils.core import setup, Extension

path = '../../../../src/numerical/random/uniform/'
setup(name='numerical',
      ext_modules=[Extension('numerical', 
                             [path + 'DiscreteUniformGeneratorMt19937.i'],
                             swig_opts=['-c++'])])
