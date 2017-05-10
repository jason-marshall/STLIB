# CycleForward.py
# A forward cycle set of reactions.  
# Usage:
# python CycleForward.py size r reactions.txt
# - size is the number of species.
# - r is the number of reactions per species.

import sys

errorMessage = \
"""Usage:
python CycleForward.py size r reactions.txt
- size is the number of species and also the number of reactions.
- r is the number of reactions per species."""

if len(sys.argv) != 4:
  print errorMessage
  raise "Wrong number of command line arguments.  Exiting..."

# Get the number of species.
size = eval(sys.argv[1])
if size < 2:
  print "You specified a size of %d.\n" % size
  raise "Error: Bad size.  Exiting..."

# Get the number of reactions per species.
reactions = eval(sys.argv[2])
if reactions < 1 or reactions > size - 1:
  print "You specified %d.\n" % reactions
  raise "Error: Bad number of reactions per species.  Exiting..."

# Write the reactions.
print "Writing the reactions file..."
outFile = open(sys.argv[3], "w")
# The number of reactions.
outFile.write("%d\n\n" % (size * reactions))
# For each species.
for m in range(0, size):
  # For each reaction for that species.
  for n in range(1, reactions + 1):
    # Reactants.
    outFile.write("1 %d 1\n" % m)
    # Products.
    outFile.write("1 %d 1\n\n" % ((m + n) % size))
outFile.close()
print "Done."
