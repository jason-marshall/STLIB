# decayingDimerizing.py
# Decaying-dimerizing reactions.
# Usage:
# python decayingDimerizing.py multiplicity reactions.txt rates.txt populations.txt
# - multiplicity How many times the reactions/species are duplicated.

import sys

errorMessage = \
"""Usage:
python decayingDimerizing.py multiplicity reactions.txt rates.txt populations.txt
- multiplicity How many times the reactions/species are duplicated."""

if len(sys.argv) != 5:
  print errorMessage
  raise "Wrong number of command line arguments.  Exiting..."

# Get the multiplicity.
multiplicity = eval(sys.argv[1])
if multiplicity < 1:
  print "You specified a multiplicity of %d." % numberOfSpecies
  raise "Error: Bad multiplicity.  Exiting..."

# Write the reactions.
print "Writing the reactions file..."
outFile = open(sys.argv[2], "w")
# The number of reactions.
outFile.write("%d\n\n" % (4 * multiplicity))
for n in range(0, multiplicity):
  i = 3 * n
  j = i + 1
  k = j + 1
  #
  # S_i -> 0
  #
  # Reactants.
  outFile.write("1 %d 1\n" % i)
  # Products.
  outFile.write("0\n\n")
  #
  # 2 S_i -> S_j
  #
  # Reactants.
  outFile.write("1 %d 2\n" % i)
  # Products.
  outFile.write("1 %d 1\n\n" % j)
  #
  # S_j -> 2 S_i
  #
  # Reactants.
  outFile.write("1 %d 1\n" % j)
  # Products.
  outFile.write("1 %d 2\n\n" % i)
  #
  # S_j -> S_k
  #
  # Reactants.
  outFile.write("1 %d 1\n" % j)
  # Products.
  outFile.write("1 %d 1\n\n" % k)
outFile.close()
print "Done."

# Write the rate constants.
print "Writing the rate constants file..."
outFile = open(sys.argv[3], "w")
# The number of reactions.
outFile.write("%d\n" % (4 * multiplicity))
# The rate constants.
for n in range(0, multiplicity):
  outFile.write("1\n0.002\n0.5\n0.04\n")
outFile.close()
print "Done."

# Write the populations.
print "Writing the populations file..."
outFile = open(sys.argv[4], "w")
# The number of species.
outFile.write("%d\n" % (3 * multiplicity))
# For each species.
for n in range(0, multiplicity):
  outFile.write("100000\n0\n0\n")
outFile.close()
print "Done."
