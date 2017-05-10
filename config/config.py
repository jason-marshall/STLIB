# This module must be called from the parent directory.

import os
import subprocess
import os.path
from collections import OrderedDict

__baseDir = os.path.dirname(os.path.realpath(__file__))
__buildDir = os.path.join(__baseDir, 'build')

# CONTINUE REMOVE Now I assume that the compiler supports SSE4.
# First check if the compiler supports SSE4.
#process = subprocess.Popen(['g++', '--target-help'],
#                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#(stdout, stderr) = process.communicate()
#sse4 = stdout.find('-msse4 ') != -1

def __hasFeature(feature):
    """Check if the processor supports the feature."""
    if not os.access(__buildDir, os.F_OK):
        os.mkdir(__buildDir)
    source = os.path.join(__baseDir, feature + '.cc')
    target = os.path.join(__buildDir, feature)
    if not os.access(target, os.F_OK):
        subprocess.check_call(['c++', '-o', target, source])
    return subprocess.call([target]) == 0

__names = OrderedDict((('avx', 'avx'), ('sse4.2', 'sse4_2'),
                       ('sse4.1', 'sse4_1')))
sse4_1 = __hasFeature('sse4_1')
sse4_2 = __hasFeature('sse4_2')
avx = __hasFeature('avx')

def addSimdFlags(env, features=__names.keys()):
    for feature in features:
        if eval(__names[feature]):
            env.AppendUnique(CCFLAGS=['-m' + feature])
            break

