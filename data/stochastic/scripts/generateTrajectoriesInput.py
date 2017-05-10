"""Generate an input file for the generateTrajectories.py script. The 
reactions are:
a -> b
b -> a"""

import sys
import string

# The model.

# initialAmounts (populations)
sys.stdout.write('100 100\n')

# packedReactions
sys.stdout.write('1 0 1 1 1 1 1 1 1 1 0 1\n')

# propensityFactors
sys.stdout.write('1.0 1.0\n')

# RNG state.

# mt19937State Use [0..623]
sys.stdout.write('%s\n' % string.join(map(repr, range(624)), ' '))

# Simulation parameters.

# method is an integer index.
sys.stdout.write('0\n')

# options is an integer index.
sys.stdout.write('0\n')

# maximumAllowedSteps
sys.stdout.write('0\n')

# frameTimes
sys.stdout.write('%s\n' % string.join(map(repr, range(11)), ' '))

# numberOfTrajectories
sys.stdout.write('2\n')

