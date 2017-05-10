# -*- python -*-

import subprocess
from SCons.Script import *

def cpuCount():
    """Return the number of available cores."""
    try:
        # The multiprocessing module was introduced in Python 2.6. It has been
        # backported 2.5 and 2.4 and is included in some distributions of these.
        import multiprocessing
        return multiprocessing.cpu_count()
    except:
        return 1

# Set the number of concurrent jobs to the number of available cores.
SetOption('num_jobs', cpuCount())

mode = ARGUMENTS.get('mode', 'release')

serial = Environment(CPPPATH='../src')
if serial['CXX'] == 'g++':
    if mode == 'release':
        # Increase the inline limit from 600 to 6000.
        serial.AppendUnique(CCFLAGS=['-O3', '-funroll-loops',
                                     '-fstrict-aliasing',
                                     '-finline-limit=6000'])
    elif mode == 'debug':
        serial.AppendUnique(CCFLAGS=['-g', '-DSTLIB_DEBUG'])
    else:
        print('Error: Expected "debug" or "release" for the mode argument, '\
              'found: "' + mode + '".')
        Exit(1)
    # Warnings. I don't use the pedantic flag because it will break code that
    # uses long long.
    serial.AppendUnique(CCFLAGS=['-ansi', '-Wall', '-Wextra',
                                 '-Wstrict-aliasing=2', '-Wno-unknown-pragmas'])
    # Use the rt library on linux, but not on darwin or windows.
    if serial['PLATFORM'] == 'posix':
        serial.AppendUnique(LIBS=['rt'])
    # We need this to compile the implicit tau-leaping solvers for PPC
    # architectures.
    serial.AppendUnique(CCFLAGS=['-DEIGEN_DONT_VECTORIZE'])
    # Architectures.
    uname = os.uname()
    if uname[0] == 'Darwin':
        # Snow Leopard is 10.x.x. It only supports Intel architectures.
        if int(uname[2].split('.')[0]) >= 10:
            arch = ['-arch', 'i386', '-arch', 'x86_64']
        else:
            # Leopard may run on either Power PC or Intel architectures.
            arch = ['-arch', 'i386', '-arch', 'x86_64', '-arch', 'ppc',
                    '-arch', 'ppc64']
        serial.Append(CCFLAGS=arch, LINKFLAGS=arch)
