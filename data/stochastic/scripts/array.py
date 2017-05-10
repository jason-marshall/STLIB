# array.py
# An array whose values are a function of the index.

import sys
from math import *
from random import *

errorMessage = \
"""Usage:
python array.py size function format array.txt
- size is the size of the array.
- function is a function of the index.
- format is the character for the output format.
- array.txt is the output array."""

if len(sys.argv) != 5:
  print errorMessage
  raise "Wrong number of command line arguments.  Exiting..."

# Get the size of the array
size = eval(sys.argv[1])
if size <= 0:
  print "You specified a size of %d\n" % size
  raise "Error: Bad size. Exiting..."

# Get the function
function = lambda n: eval(sys.argv[2])

# Get the format.
format = "%" + sys.argv[3] + "\n"

# Write the reactions.
print "Writing the array..."
outFile = open(sys.argv[4], "w")
# The size.
outFile.write("%d\n" % size)
# The array elements
for n in range(size):
  outFile.write(format % function(n))
outFile.close()
print "Done."
