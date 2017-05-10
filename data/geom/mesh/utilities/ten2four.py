# ten2four.py
# Convert a 10 node tetrahedral mesh to a 4 node tetrahedral mesh.
# This script only drops the midpoint node indices from the connectivities
# list.  It does not remove the midpoint nodes from the node location list.
# Use pack.py for that.
#
# Usage:
# python ten2four.py mesh10 mesh4

import sys

if len(sys.argv) != 3:
    print("Usage:")
    print("python ten2four.py mesh10 mesh4\n")
    print("Bad command line arguments.  Exiting...")
    sys.exit()

# Open the files.
assert sys.argv[1] != sys.argv[2]
inFile = open(sys.argv[1], "r")
outFile = open(sys.argv[2], "w")

#
# The space dimension and the simplex dimension.
#
numberStrings = inFile.readline().split()
spaceDimension = int(numberStrings[0])
simplexDimension = int(numberStrings[1])
print("space dimension = %d, simplex dimension = %d" % (spaceDimension, 
                                                        simplexDimension))
outFile.write("%d 3\n" % (spaceDimension))

#
# Read the nodes.
#
numNodes = int(inFile.readline())
print("Reading and writing the %d node locations..." % numNodes)
outFile.write("%d\n" % numNodes)
for n in range(numNodes):
    outFile.write(inFile.readline())
print("Done.")

#
# Read the elements.
#
# The corner indices.
#c = (0, 2, 4, 9)
c = (0, 1, 2, 3)
numElements = int(inFile.readline())
print("Extracting the corner nodes for the %d elements...." % numElements)
outFile.write("%d\n" % numElements)
for n in range(numElements):
    numberStrings = inFile.readline().split()
    # Extract the corner nodes.
    outFile.write("%s %s %s %s\n" % (numberStrings[c[0]], numberStrings[c[1]],
                                     numberStrings[c[2]], numberStrings[c[3]]))
print("Done.")
