# ProdDegDimDiff.py
# Usage:
# python ProdDegDimDiff.py multiplicity

# This model adds diffusion to Lipshtat's production, degredation, and
# dimerization model. The dimerized species are omitted. Thus there are five
# reaction for each species:
# Production, 0 -> S_i
# Degredation, S_i -> 0
# Dimerization, 2 S_i -> 0
# Diffusion left, S_i -> S_{(i-1)%N}
# Diffusion right, S_i -> S_{(i+1)%N}

# The output is
# <number of species>
# <number of reactions>
# <list of initial amounts>
# <packed reactions>
# <list of propensity factors>

import sys
import random

def output(multiplicity):
    s = []
    # <number of species>
    s.append('%d\n' % multiplicity)
    # <number of reactions>
    s.append('%d\n' % (5 * multiplicity))
    # <list of initial amounts>
    s.append('0 ' * multiplicity + '\n')
    # <packed reactions>
    for n in range(multiplicity):
        # Production, 0 -> S_i
        s.append('0 1 ' + str(n) + ' 1 ')
        # Degredation, S_i -> 0
        s.append('1 ' + str(n) + ' 1 0 ')
        # Dimerization, 2 S_i -> 0
        s.append('1 ' + str(n) + ' 2 0 ')
        # Diffusion left, S_i -> S_{(i-1)%N}
        s.append('1 ' + str(n) + ' 1 1 ' + str((n-1)%multiplicity) + ' 1 ')
        # Diffusion right, S_i -> S_{(i+1)%N}
        s.append('1 ' + str(n) + ' 1 1 ' + str((n+1)%multiplicity) + ' 1 ')
    s.append('\n')
    # <list of propensity factors>
    s.append('5 2 4 2 2 ' * multiplicity + '\n')
    return ''.join(s)

if __name__ == '__main__':
    errorMessage = \
    """Usage:
    python ProdDegDimDiff.py multiplicity"""

    if len(sys.argv) != 2:
        print errorMessage
        raise "Wrong number of command line arguments.  Exiting..."

    # Get the multiplicity.
    multiplicity = int(sys.argv[1])
    assert multiplicity > 0

    print output(multiplicity)
