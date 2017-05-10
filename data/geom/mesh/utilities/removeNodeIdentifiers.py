"""removeNodeIdentifiers.py
This script converts a mesh in the format

3 2
numNodes
id x y z
...
numSimplices
i j k

to the standard format. Usage:
python removeNodeIdentifiers.py input output
"""
import sys, string

if len(sys.argv) != 3:
    print "Usage:"
    print "python removeNodeIdentifiers.py input output"

# Open the files.
inFile = open(sys.argv[1], "r")
outFile = open(sys.argv[2], "w")

#
# The space dimension and the cell dimension.
#
numberStrings = string.split(inFile.readline())
spaceDimension = int(numberStrings[0])
cellDimension = int(numberStrings[1])
print "space dimension = %d, cell dimension = %d" % (spaceDimension, 
                                                     cellDimension)
outFile.write("%d %d\n" % (spaceDimension, cellDimension))

#
# The vertices.
#
numberOfVertices = int(inFile.readline())
print "%d vertices." % numberOfVertices
outFile.write("%d\n" % numberOfVertices)


print "Reading and writing the vertices..."
identifiersToIndex = {}
for n in range(numberOfVertices):
    line = inFile.readline()
    identifiersToIndex[int(line.split()[0])] = n
    for coordinate in line.split()[1:]:
        outFile.write("%s " % coordinate)
    outFile.write("\n")
print "Done."

#
# The indexed cells.
#
numberOfCells = int(inFile.readline())
print "%d cells." % numberOfCells
outFile.write("%d\n" % numberOfCells)

print "Reading and writing the cells..."
for n in range(numberOfCells):
    for identifier in inFile.readline().split():
        outFile.write("%d " % identifiersToIndex[int(identifier)])
    outFile.write("\n")
print "Done."

inFile.close()
outFile.close()
