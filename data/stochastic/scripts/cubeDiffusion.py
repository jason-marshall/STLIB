# cubeDiffusion.py
# An all-to-all set of reactions.  
# Usage:
# python cubeDiffusion.py cellsPerDimension population rate reactions.txt
# populations.txt
# - cellPerDimension is the number of cells in each dimension.
# - population is the population.  (The same for each species.)
# - rate is the rate constant.  (The same for each reaction.)

import sys

errorMessage = \
"""Usage:
python cubeDiffusion.py cellsPerDimension population rate reactions.txt
populations.txt
- cellPerDimension is the number of cells in each dimension.
- population is the population.  (The same for each species.)
- rate is the rate constant.  (The same for each reaction.)"""

if len(sys.argv) != 6:
  print errorMessage
  raise "Wrong number of command line arguments.  Exiting..."

# Get the number of cells per dimension.
cellsPerDimension = eval(sys.argv[1])
if cellsPerDimension < 2:
  print "You specified " << cellsPerDimension << " cells per dimension."
  raise "Error: This parameter must be at least 2.  Exiting..."

# Get the population of all species.
population = eval(sys.argv[2])

# Get the rate constant of all reactions.
rate = eval(sys.argv[3])

# Function for computing the species index from the grid indices.
def computeIndex(i, j, k):
  return i + j * cellsPerDimension + k * cellsPerDimension * cellsPerDimension 

# Function for writing a reaction.
def writeReaction(i, j):
  # Reactants.
  outFile.write("0 1 %d 1\n" % i)
  # Products.
  outFile.write("0 1 %d 1\n" % j)
  # Rate constant.
  outFile.write("%.16g\n\n" % rate)  

# Write the reactions.
print "Writing the reactions file..."
outFile = open(sys.argv[4], "w")
# The number of reactions.
numberOfReactions = (cellsPerDimension * cellsPerDimension *
                     (cellsPerDimension - 1) * 3 * 2)
outFile.write("%d\n\n" % numberOfReactions)
# For each reaction.
for i in range(0, cellsPerDimension):
  for j in range(0, cellsPerDimension):
    for k in range(0, cellsPerDimension):
      if i != cellsPerDimension - 1:
        writeReaction(computeIndex(i, j, k), computeIndex(i + 1, j, k))
      if i != 0:
        writeReaction(computeIndex(i, j, k), computeIndex(i - 1, j, k))
      if j != cellsPerDimension - 1:
        writeReaction(computeIndex(i, j, k), computeIndex(i, j + 1, k))
      if j != 0:
        writeReaction(computeIndex(i, j, k), computeIndex(i, j - 1, k))
      if k != cellsPerDimension - 1:
        writeReaction(computeIndex(i, j, k), computeIndex(i, j, k + 1))
      if k != 0:
        writeReaction(computeIndex(i, j, k), computeIndex(i, j, k - 1))

outFile.close()
print "Done."

# Write the populations.
print "Writing the populations file..."
outFile = open(sys.argv[5], "w")
# The number of species.
numberOfSpecies = cellsPerDimension * cellsPerDimension * cellsPerDimension
outFile.write("%d\n" % numberOfSpecies)
# For each species.
for n in range(0, numberOfSpecies):
  outFile.write("%d\n" % population)
outFile.close()
print "Done."
