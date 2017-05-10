# -*- python -*-

import re
import os
import os.path
import platform
import subprocess
from SCons.Script import *
import sys

# SCons renames the pickle module as cPickle. This prevents me from creating
# the lock variable. The following as a hack to work around this.
# http://stackoverflow.com/questions/24453387/scons-attributeerror-builtin-function-or-method-object-has-no-attribute-disp
del sys.modules['pickle']
del sys.modules['cPickle']
import multiprocessing
mpiLock = multiprocessing.Lock()
import pickle
import cPickle

#sys.path.append('../..')
#from config.config import addSimdFlags

AddOption('--optimize', action='store_true', dest='optimize')

# On Windows, serial['CXX'] is '$CC'. That is, it just redirects to the C
# compiler.
useMsvc = Environment()['CC'] == 'cl'

def run(target, source, env, prefix):
    process = subprocess.Popen(prefix + os.path.join('.', str(source[0])),
                               shell=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=env['ENV'])
    (stdout, stderr) = process.communicate()
    file = open(str(target[0]), 'w')
    # If the process dies with a segmentation fault it may not write anything
    # to stderr.
    if process.returncode != 0:
        file.write('Terminated with code ' + str(process.returncode) + '.\n')
    file.write(stderr)
    return None

def runSerial(target, source, env):
    run(target, source, env, '')

# Count physical instead of logical processors. (Assume hyperthreading 
# overcounts by a factor of two.)
mpiNumProcesses = max(multiprocessing.cpu_count() / 2, 2)
# Find the mpirun command.
if useMsvc:
    mpiRun = 'mpiexec /np %d ' % mpiNumProcesses
else:
    for name in ('mpirun', 'openmpirun'):
        # When I can assume Python 2.7, use the following in a try statement.
        # mpiRun = subprocess.check_output(['which', name]).strip()
        # For now, I need to use Popen.
        try:
            p = subprocess.Popen(['which', name], stdout=subprocess.PIPE)
            output = p.communicate()[0].strip()
        except:
            output = ''
        if output:
            mpiRun = output + ' -np %d ' % mpiNumProcesses
            break


def runMpi(target, source, env):
    # Only run one MPI test at a time.
    mpiLock.acquire()
    run(target, source, env, mpiRun)
    mpiLock.release()

    
def report(target, source, env):
    errors = ''
    numErrors = 0
    for s in source:
        error = open(str(s), 'r').read()
        if error:
            numErrors += 1
            errors += '\n' + str(s).split('.')[0] + ':\n' + error
    if numErrors == 0:
        print('\nWoohoo! All ' + str(len(source)) + ' test suites succeeded.\n')
    else:
        print('\nDoh! ' + str(numErrors) + ' of the ' + str(len(source)) +
              ' test suites failed.')
        print(errors)
        # Exit with a nonzero return value so that it is apparent that errors 
        # were encountered.
        Exit(1)
    return None


# Use the external environment.
# Note that SCons will not add directories in the environment variable
# CPLUS_INCLUDE_PATH to the include path. ~/include is a popular place to
# put headers.
serial = Environment(ENV=os.environ,
                     BUILDERS=
                     {'Run':Builder(action=runSerial, suffix='.err',
                                    src_suffix=Environment().Dictionary()
                                    ['PROGSUFFIX']),
                      'Report':Builder(action=report, src_suffix='.err')},
                     CPPPATH=[os.path.realpath('../..'),
                              os.path.expanduser('~/include')])

if 'CXX' in ARGUMENTS:
    serial['CXX'] = ARGUMENTS['CXX']
else:
    # For OS X 10.9 (Mavericks) and beyond, use clang++ by default.
    if platform.system() == 'Darwin' and int(os.uname()[2].split('.')[0]) >= 13:
        serial['CXX'] = 'clang++'

useClang = serial['CXX'].find('clang') != -1
useGcc = not useClang and serial['CXX'].find('g++') != -1
# We use GCC-style flags for GCC and clang.
useGccFlags = useGcc or useClang

# Options for MSVC.
if useMsvc:
    # /Zi enables debugging (/DEBUG) and the generation of debugging (.pdb)
    # files. However, I can't get the debugger to work, so the option isn't 
    # useful.
    serial.AppendUnique(CCFLAGS=['/EHsc', '/DSTLIB_DEBUG'])

# Options that are common to GCC and Clang.
if useGccFlags:
    # Options that are common to GCC and Clang.
    if GetOption('optimize'):
        serial.AppendUnique(CCFLAGS=['-O3', '-funroll-loops'])
    else:
        serial.AppendUnique(CCFLAGS=['-g', '-DSTLIB_DEBUG'])
    # Version 0 refers to the version conforming most closely to the C++
    # ABI specification. Therefore, the ABI obtained using version 0 will
    # change in different versions of GCC as ABI bugs are fixed.
    # One must use either version 0 or version 6 in order to overload functions
    # using __m128 and __m256.
    # I don't use the pedantic flag because it will break code that uses 
    # long long. Don't use -ansi because it causes problems with std::tuple.
    serial.AppendUnique(CCFLAGS=['-std=c++11', '-Wall', '-Wextra', '-Werror',
                                 '-Wstrict-aliasing=2', '-Wno-unknown-pragmas'])
    # Use the rt library on linux, but not on darwin or windows.
    if serial['PLATFORM'] == 'posix':
        serial.AppendUnique(LIBS=['rt'])
    
if useGcc:
    serial.AppendUnique(CCFLAGS=['-Wno-pragmas', '-fabi-version=0'])

if useClang:
    serial.AppendUnique(CCFLAGS=['-Woverloaded-virtual'])
    serial.AppendUnique(CCFLAGS=['-stdlib=libc++'],
                        LINKFLAGS=['-stdlib=libc++'])

    
# SIMD specializations.
sse2 = serial.Clone()
if useGccFlags:
    sse2.AppendUnique(CCFLAGS=['-msse2'])
avx = serial.Clone()
if useGccFlags:
    avx.AppendUnique(CCFLAGS=['-mavx'])
if useGccFlags:
    # For serial, the following option will add the appropriate SIMD flags.
    serial.AppendUnique(CCFLAGS=['-march=native'])

openMP = serial.Clone()
if useGccFlags:
    openMP.AppendUnique(CCFLAGS=['-fopenmp'],
                        LINKFLAGS=['-fopenmp', '-L/usr/local/lib/x86_64'])
    
mpi = None
if useMsvc:
    mpi = Environment(
        ENV=os.environ,
        BUILDERS={'Run':Builder(action=runMpi, suffix='.err',
                                src_suffix=Environment().Dictionary()
                                ['PROGSUFFIX']),
                  'Report':Builder(action=report, src_suffix='.err')},
        CPPPATH=[os.path.realpath('../..'),
                 os.path.expanduser('~/include'),
                 r'C:\Program Files\Microsoft SDKs\MPI\Include'],
        LIBS=['msmpi'],
        LIBPATH=[r'C:\Program Files\Microsoft SDKs\MPI\Lib\x86'])
    mpi.AppendUnique(CCFLAGS=['/EHsc', '/DSTLIB_DEBUG'])
else:
    for name in ('mpic++', 'openmpic++'):
        # When I can assume Python 2.7, use the following in a try statement.
        # cxx = subprocess.check_output(['which', name]).strip()
        # For now, I need to use Popen.
        try:
            p = subprocess.Popen(['which', name], stdout=subprocess.PIPE)
            cxx = p.communicate()[0].strip()
        except:
            cxx = ''
        if not cxx:
            continue

        mpi = Environment(
            ENV=os.environ,
            BUILDERS={'Run':Builder(action=runMpi, suffix='.err',
                                    src_suffix=Environment().Dictionary()
                                    ['PROGSUFFIX']),
                      'Report':Builder(action=report, src_suffix='.err')},
            CXX=cxx,
            CPPPATH=[os.path.realpath('../..'),
                     os.path.expanduser('~/include')])
        mpi.AppendUnique(CCFLAGS=['-g', '-DSTLIB_DEBUG'])
        mpi.AppendUnique(
            CCFLAGS=['-std=c++11', '-Wall', '-Wextra', '-Werror',
                     '-Wstrict-aliasing=2', '-Wno-unknown-pragmas'])
        if useGcc:
            mpi.AppendUnique(CCFLAGS=['-Wno-pragmas', '-fabi-version=0'])
        # Use the rt library on linux, but not on darwin or windows.
        if mpi['PLATFORM'] == 'posix':
            mpi.AppendUnique(LIBS=['rt'])
            mpi.AppendUnique(CCFLAGS=['-fopenmp'], LINKFLAGS=['-fopenmp'])
        #if mpi['PLATFORM'] == 'darwin':
        #    mpi['ENV']['OMPI_CXX'] = '/opt/local/bin/g++-mp-4.8'
        break
if mpi is None:
    print('Warning: MPI is not available.')

   
cudart = Environment(ENV=os.environ,
                     CC='/Developer/NVIDIA/CUDA-5.5/bin/nvcc',
                     CPPPATH=os.path.realpath('../..'))
cudart.Append(CCFLAGS=['-ccbin', '/usr/bin/clang',
                       '-m64',
                       '-Xcompiler', '-arch', '-Xcompiler', 'x86_64',
                       '-Xcompiler', '-stdlib=libstdc++'],
            LINKFLAGS=['-ccbin', '/usr/bin/clang',
                       '-m64',
                       '-Xcompiler', '-arch', '-Xcompiler', 'x86_64',
                       '-Xcompiler', '-stdlib=libstdc++',
                       '-Xlinker', '-rpath',
                       '-Xlinker', '/Developer/NVIDIA/CUDA-5.5/lib'])

# CONTINUE -O2?
cudaObject = Builder(action='$CC -ccbin /usr/bin/clang -m64 -Xcompiler -arch -Xcompiler x86_64 -Xcompiler -stdlib=libstdc++ -I$CPPPATH -c $SOURCE -o $TARGET')
cuda = Environment(ENV=os.environ,
                   CC='/Developer/NVIDIA/CUDA-5.5/bin/nvcc',
                   BUILDERS={'CudaObject': cudaObject},
                   CPPPATH=os.path.realpath('../..'))
