# -*- python -*-

import re
import subprocess
from SCons.Script import *

def run(target, source, env):
    process = subprocess.Popen('python ' + str(source[0]), shell=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=env['ENV'])
    (stdout, stderr) = process.communicate()
    if stderr:
        print(stderr)
        Exit(1)
    # If the process dies with a segmentation fault it may not write anything
    # to stderr.
    if process.returncode != 0:
        print('Terminated with code ' + str(process.returncode) + '.\n')
        Exit(1)
    return None

serial = Environment(BUILDERS=
                     {'Run':Builder(action=run, suffix='.err', src_suffix='')})
