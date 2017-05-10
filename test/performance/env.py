# -*- python -*-

import os
import platform
import sys
import subprocess
from SCons.Script import *

# Use "scons mode=debug" for debug mode.
mode = ARGUMENTS.get('mode', 'release')
modes = ('release', 'profile', 'debug')
if not mode in modes:
    print('Error: Invalid mode. Valid choices are: ' + ', '.join(modes) + '.')
    Exit(1)

# Serial.
# Note that SCons will not add directories in the environment variable
# CPLUS_INCLUDE_PATH to the include path. ~/include is a popular place to
# put headers.
serial = Environment(ENV=os.environ,
                     CPPPATH=[os.path.realpath('../..'),
                              os.path.expanduser('~/include')])

if 'CXX' in ARGUMENTS:
    serial['CXX'] = ARGUMENTS['CXX']
else:
    # For OS X 10.9 (Mavericks) and beyond, use clang++ by default.
    if platform.system() == 'Darwin' and int(os.uname()[2].split('.')[0]) >= 13:
        serial['CXX'] = 'clang++'

useGcc = serial['CXX'].find('g++') == 0
# We use GCC-style flags for GCC and clang.
useGccFlags = useGcc or serial['CXX'] == 'clang++'
# On Windows, serial['CXX'] is '$CC'. That is, it just redirects to the C
# compiler.
useMsvc = serial['CC'] == 'cl'

# Options for MSVC.

if useMsvc:
    serial.AppendUnique(CCFLAGS=['/EHsc'])

# Options that are common to GCC and Clang.
if useGccFlags:
    if mode == 'release':
        serial.AppendUnique(
            CCFLAGS=['-O3', '-funroll-loops', '-DNDEBUG'])
    elif mode == 'profile':
        # Zoom recomends -fno-omit-frame-pointer for linux kernel 2.6.32.
        serial.AppendUnique(
            CCFLAGS=['-g', '-O3', '-funroll-loops', '-fno-omit-frame-pointer'])
    else:
        serial.AppendUnique(
            CCFLAGS=['-g', '-DSTLIB_DEBUG'])
    # I don't use the pedantic flag because it will break code that uses 
    # long long. Don't use -ansi because it causes problems with std::tuple.
    # I need c++0x for std::array and std::tuple.
    serial.AppendUnique(
        CCFLAGS=['-std=c++0x', '-Wall', '-Wextra', '-Werror',
                 '-Wstrict-aliasing=2', '-Wno-unknown-pragmas'])
    # Use the rt library on linux, but not on darwin.
    if serial['PLATFORM'] == 'posix':
        serial.AppendUnique(LIBS=['rt'])

# Increase the inline limit from 600 to 6000.
#serial.AppendUnique(CCFLAGS=['-finline-limit=6000'])

if useGcc:
    serial.AppendUnique(CCFLAGS=['-Wno-pragmas', '-fabi-version=0'])

# SIMD specializations.
scalar = serial.Clone()
scalar.AppendUnique(CCFLAGS=['-DSTLIB_NO_SIMD_INTRINSICS'])

if False:
    # The cpu dispatch environment may not have SIMD options.
    cpuDispatch = serial.Clone()
    sse2 = serial.Clone()
    sse2.AppendUnique(CCFLAGS=['-msse2', '-mno-avx'])
    avx = serial.Clone()
    avx.AppendUnique(CCFLAGS=['-mavx'])

# For serial, the -march option will add the appropriate SIMD flags.
# Use "scons march=ARCH" to use a specific instruction set. By default,
# we compile for the native architecture.
march = ARGUMENTS.get('march', 'native')
if useGccFlags:
    serial.AppendUnique(CCFLAGS=['-march=' + march])

if False:
    openCL = serial.Clone()
    if useGccFlags:
        openCL.AppendUnique(LINKFLAGS=['-framework', 'OpenCL'])

openMP = serial.Clone()
if useGcc:
    # CONTINUE: clang does not yet support openMP. It will accept the -openmp
    # flag, but then won't generate output.
    openMP.AppendUnique(CCFLAGS=['-fopenmp'], LINKFLAGS=['-fopenmp'])
    if False:
        openMP.AppendUnique(CCFLAGS=['-fopenmp'],
                            LINKFLAGS=['-fopenmp', '-L/usr/local/lib/x86_64'])

cudart = Environment(ENV=os.environ,
                     CXX='/Developer/NVIDIA/CUDA-5.5/bin/nvcc',
                     CC='/Developer/NVIDIA/CUDA-5.5/bin/nvcc',
                     CPPPATH=os.path.realpath('../..'))
cudart.Append(CCFLAGS=['-ccbin', '/usr/bin/clang',
                       '-m64', '-O2',
                       '-Xcompiler', '-arch', '-Xcompiler', 'x86_64',
                       '-Xcompiler', '-stdlib=libstdc++'],
# This doesn't help for std::array.
#                       '-Xcompiler', '-std=c++0x'
            LINKFLAGS=['-ccbin', '/usr/bin/clang',
                       '-m64',
                       '-Xcompiler', '-arch', '-Xcompiler', 'x86_64',
                       '-Xcompiler', '-stdlib=libstdc++',
                       '-Xlinker', '-rpath',
                       '-Xlinker', '/Developer/NVIDIA/CUDA-5.5/lib'])

cudaObject = Builder(action='$CC -ccbin /usr/bin/clang -m64 -O2 -Xcompiler -arch -Xcompiler x86_64 -Xcompiler -stdlib=libstdc++ -I$CPPPATH -c $SOURCE -o $TARGET')
cuda = Environment(ENV=os.environ,
                   CC='/Developer/NVIDIA/CUDA-5.5/bin/nvcc',
                   BUILDERS={'CudaObject': cudaObject},
                   CPPPATH=os.path.realpath('../..'))

mpi = None
if useMsvc:
    mpi = Environment(
        ENV=os.environ,
        CPPPATH=[os.path.realpath('../..'),
                 os.path.expanduser('~/include'),
                 r'C:\Program Files\Microsoft SDKs\MPI\Include'],
        LIBS=['msmpi'],
        LIBPATH=[r'C:\Program Files\Microsoft SDKs\MPI\Lib\x86',
                 r'C:\local\boost_1_58_0\lib32-msvc-12.0'])
    mpi.AppendUnique(CCFLAGS=['/EHsc'])
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
            CXX=cxx,
            CPPPATH=[os.path.realpath('../..'),
                     os.path.expanduser('~/include')])

        if mode == 'release':
              mpi.AppendUnique(CCFLAGS=['-O3', '-funroll-loops', '-DNDEBUG'])
        elif mode == 'profile':
              mpi.AppendUnique(CCFLAGS=['-g', '-O3', '-funroll-loops'])
        else:
            mpi.AppendUnique(CCFLAGS=['-g', '-DSTLIB_DEBUG'])
        mpi.AppendUnique(CCFLAGS=['-march=' + march])
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
