# unit.py
# Write a probability mass function with unit values.  The size of the PMF
# array is a command line argument.

import sys

size = int(sys.argv[1])
print size
for i in range(size):
    print "1"
