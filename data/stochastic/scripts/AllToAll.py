# AllToAll.py
# An all-to-all set of reactions.  

import sys

errorMessage = \
"""Usage:
python AllToAll.py numberOfSpecies reactions.txt
- numberOfSpecies is the number of species."""

if len(sys.argv) != 3:
  print errorMessage
  raise "Wrong number of command line arguments.  Exiting..."

# Get the number of species.
numberOfSpecies = eval(sys.argv[1])
if numberOfSpecies < 2:
  print "You specified " << numberOfSpecies << "species."
  raise "Error: Bad number of species.  Exiting..."

# Write the reactions.
print "Writing the reactions file..."
outFile = open(sys.argv[2], "w")
# The number of reactions.
outFile.write("%d\n\n" % (numberOfSpecies * (numberOfSpecies - 1)))
# For each reaction.
for m in range(0, numberOfSpecies):
  for n in range(0, numberOfSpecies):
    if m != n:
      # Reactants.
      outFile.write("1 %d 1\n" % m)
      # Products.
      outFile.write("1 %d 1\n" % n)
outFile.close()
print "Done."
