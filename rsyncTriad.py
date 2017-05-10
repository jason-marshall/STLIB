# Clone stlib and then run this script to sync triad with it. The argument
# is the path to where triad is checked out.

import os
import os.path
import subprocess
from optparse import OptionParser

parser = OptionParser()
(options, args) = parser.parse_args()
# If the path to Triad is not specified, assume it is '..'.
if args:
    path = args[0]
    del args[0]
else:
    path = '..'
assert not args
path = os.path.join(path, 'triad')
assert os.access(path, os.F_OK)

for top in ['doc/', 'src/', 'test/unit/']:
    for directory in ['ads/', 'array/', 'ext/', 'geom/', 'levelSet/', 'mst/',
                      'numerical/']:
        command = 'rsync -a ' + os.path.join(top, directory) + ' ' \
                  + os.path.join(path, top, directory)
        print(command)
        subprocess.check_call(command, shell=True)

for top in ['src/', 'test/unit/']:
    for directory in ['cuda/', 'hj/', 'simd/']:
        command = 'rsync -a ' + os.path.join(top, directory) + ' ' \
                  + os.path.join(path, top, directory)
        print(command)
        subprocess.check_call(command, shell=True)

# Get rid of the geom/mesh graphics files.
command = 'rm -rf ' + os.path.join(path, 'doc', 'geom/graphics/*.*')
print(command)
subprocess.check_call(command, shell=True)
command = 'rm -rf ' + os.path.join(path, 'doc', 'geom/graphics/billiard')
print(command)
subprocess.check_call(command, shell=True)

# Eigen
command = 'rsync -a src/Eigen/ ' + os.path.join(path, 'src/Eigen')
print(command)
subprocess.check_call(command, shell=True)

# Select files in doc.
for x in ['SConstruct', 'doxygen.py', 'main.cfg', 'astyle.txt', 'doxypy.py']:
    command = 'rsync ' + os.path.join('doc', x)  + ' ' \
              + os.path.join(path, 'doc', x)
    print(command)
    subprocess.check_call(command, shell=True)

# Select files and directories in test/unit.
# I removed SConstruct because of customizations in Triad.
for x in ['env.py', 'num_jobs.py']:
    command = 'rsync -a ' + os.path.join('test/unit', x)  + ' ' \
              + os.path.join(path, 'test/unit', x)
    print(command)
    subprocess.check_call(command, shell=True)

# test/performance
command = 'rsync -a test/performance/ ' +\
          os.path.join(path, 'test/performance/')
print(command)
subprocess.check_call(command, shell=True)
command = 'rm -rf ' + os.path.join(path, 'test/performance/amr')
print(command)
subprocess.check_call(command, shell=True)

# Select files and directories in examples.
for directory in ['geom/', 'mst/', 'SConstruct', 'env.py', 'num_jobs.py']:
    command = 'rsync -a ' + os.path.join('examples/', directory) + ' '\
              + os.path.join(path, 'test/apps/', directory)
    print(command)
    subprocess.check_call(command, shell=True)
# Get rid of the unused directories.
command = 'rm -rf ' + os.path.join(path, 'examples/geom/spatialIndexing') + ' '\
          + os.path.join(path, 'examples/geom/mesh/experimental') + ' '\
          + os.path.join(path, 'examples/geom/mesh/optimization')
print(command)
subprocess.check_call(command, shell=True)
