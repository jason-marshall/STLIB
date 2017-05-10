#! /usr/bin/env python
"""Recurse directories and run the unit tests."""

import os
import os.path
import random
import subprocess
import sys

numSuites = 0
errors = []
# Walk over the directory tree.
for root, dirs, files in os.walk('.'):
    # Get the executables in this directory.
    executables = filter(lambda x: len(x) > 4 and x[-4:] == '.exe', files)
    # Add the path.
    executables = map(lambda x: os.path.join(root, x), executables)
    for program in executables:
        # Run the test suite.
        numSuites += 1
        process = subprocess.Popen(program, shell=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        (stdout, stderr) = process.communicate()
        if process.returncode == 0:
            sys.stdout.write('.')
        else:
            sys.stdout.write('Doh!')
            errors.append((program, stderr))
if errors:
    numFailed = len(errors)
    numSucceeded = numSuites - numFailed
    print('\n' + str(numSucceeded) + ' test suites succeeded, ' +
          str(numFailed) + ' failed.\n')
    for name, error in errors:
        print(name + ' failed.\n' + error + '\n')
else:
    print('\nWoohoo! All ' + str(numSuites) + ' test suites succeeded.')
    
