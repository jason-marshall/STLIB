# CycleBidirectional.py
# A bi-directional cycle set of reactions.  
# Usage:
# python CycleBidirectional.py size reactions.txt
# - size is the number of species.

import sys

errorMessage = \
"""Usage:
python CycleBidirectional.py size reactions.txt
- size is the number of species and also the number of reactions."""

if len(sys.argv) != 3:
  print errorMessage
  raise "Wrong number of command line arguments.  Exiting..."

# Get the number of species.
size = eval(sys.argv[1])
if size < 2:
  print "You specified a size of %d.\n" % size
  raise "Error: Bad size.  Exiting..."

# Write the reactions.
print "Writing the reactions file..."
outFile = open(sys.argv[2], "w")
# The number of reactions.
outFile.write("%d\n\n" % (2 * size))
# For each reaction.
for m in range(0, size):
  # Forward.
  # Reactants.
  outFile.write("1 %d 1\n" % m)
  # Products.
  outFile.write("1 %d 1\n\n" % ((m + 1) % size))
  # Backward.
  # Reactants.
  outFile.write("1 %d 1\n" % m)
  # Products.
  outFile.write("1 %d 1\n\n" % ((m + size - 1) % size))
outFile.close()
print "Done."
