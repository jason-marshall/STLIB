# Decay.py
# A decaying set of reactions.  
# Usage:
# python Decay.py size reactions.txt
# - size is the number of species and also the number of reactions.

import sys

errorMessage = \
"""Usage:
python Decay.py size reactions.txt
- size is the number of species and also the number of reactions."""

if len(sys.argv) != 3:
  print errorMessage
  raise "Wrong number of command line arguments.  Exiting..."

# Get the number of species.
size = int(sys.argv[1])
assert(size > 0)

# Write the reactions.
print "Writing the reactions file..."
outFile = open(sys.argv[2], "w")
# The number of reactions.
outFile.write("%d\n\n" % size)
# For each reaction.
for m in range(0, size):
  # Reactants.
  outFile.write("1 %d 1\n" % m)
  # Products.
  outFile.write("0\n\n")
outFile.close()
print "Done."
