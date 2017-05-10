# uniformRandom.py
# Write a probability mass function with uniform random values.  The size 
# of the PMF array is a command line argument.

import sys
import random

size = int(sys.argv[1])
print size
for i in range(size):
    print random.random()
