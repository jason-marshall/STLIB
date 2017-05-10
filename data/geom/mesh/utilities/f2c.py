# f2c.py
# Convert a mesh with fortran indices to one with C indices.
# i.e. subtract 1 from each index so the lowest index is 0.
# Usage:
# python f2c.py fortran_mesh c_mesh

import sys, string

if len(sys.argv) != 3:
    print "Usage:"
    print "python f2c.py fortran_mesh c_mesh\n"
    print "This program converts a mesh with fortran indices (starting at 1)"
    print "into a mesh with C indices (starting at 0)."
    print "Bad command line arguments.  Exiting..."
    sys.exit()

# Open the files.
assert sys.argv[1] != sys.argv[2]
inFile = open(sys.argv[1], "r")
outFile = open(sys.argv[2], "w")

#
# The space dimension and the cell dimension.
#
numberStrings = inFile.readline().split()
assert len(numberStrings) == 2
spaceDimension = int(numberStrings[0])
cellDimension = int(numberStrings[1])
print('space dimension = %d, cell dimension = %d' % (spaceDimension,
                                                     cellDimension))
outFile.write('%d %d\n' % (spaceDimension, cellDimension))

#
# The vertices.
#
numberOfVertices = int(inFile.readline())
print('%d vertices.' % numberOfVertices)
outFile.write('%d\n' % numberOfVertices)

print "Reading and writing the vertices..."
for n in range(numberOfVertices):
    outFile.write(inFile.readline())
print "Done."

#
# The indexed cells.
#
numberOfCells = int(inFile.readline())
print "%d cells." % numberOfCells
outFile.write("%d\n" % numberOfCells)

print "Converting fortran indices to C indices..."
for n in range(numberOfCells):
    numberStrings = inFile.readline().split()
    format = "%d" + " %d" * (len(numberStrings) - 1) + "\n"
    outFile.write(format %
                  tuple(map(lambda x: x-1, map(int, numberStrings))))
print "Done."
