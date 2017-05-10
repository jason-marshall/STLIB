import os, subprocess, sys, re
from optparse import OptionParser

parser = OptionParser()
(options, args) = parser.parse_args()
assert len(args) <= 1
# The targets may be:
# x86 for 32-bit native
# x86_amd64 for 64-bit cross
# amd64 for 64-bit native
if args:
    target = args[0]
else:
    target = ''

# Get the solver source files.
listing = os.listdir('.')
fileNames = []
for name in listing:
    base, ext = os.path.splitext(name)
    if ext == '.cc' and re.match('Homogeneous', name) and\
           not base + '.exe' in listing:
        fileNames.append(name)

# Compile the solvers.
for name in fileNames:
    if not os.access(name[:-2] + 'exe', os.F_OK):
        command = r'"vc10vars32.bat ' + target \
                  + r'"&&cl /I..\src /I..\src\third-party /Ox /EHsc ' + name
        subprocess.check_call(command)

