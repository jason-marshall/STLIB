from distutils.core import setup, Extension

import os
os.environ['CC'] = 'g++'
os.environ['CXX'] = 'g++'
os.environ['CPP'] = 'g++'

# Get the modules in this directory.
modules = []
for file in os.listdir(os.getcwd()):
    if len(file) > 2 and file[-2:] == '.i':
        modules.append(file[:-2])

# Setup the modules.
for x in modules:
    setup(name=x, ext_modules=[Extension(x, [x + '.cpp'], language='c++')])
