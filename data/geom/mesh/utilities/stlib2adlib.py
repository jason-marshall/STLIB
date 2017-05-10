# stlib2adlib.py
# Convert an stlib mesh to an adlib mesh.
# python stlib2adlib.py x.txt y.txt

import sys, string

if len(sys.argv) != 3:
    print "Usage:"
    print "python stlib2adlib.py x.txt y.txt\n"
    print "Bad command line arguments.  Exiting..."
    sys.exit()

# Open the files.
assert sys.argv[1] != sys.argv[2]
inFile = open(sys.argv[1], "r")
outFile = open(sys.argv[2], "w")

#
# Space dimension, cell dimension
#
numberStrings = inFile.readline().split()
assert len(numberStrings) == 2
spaceDimension = int(numberStrings[0])
cellDimension = int(numberStrings[1])
print('space dimension = %d, cell dimension = %d' % (spaceDimension,
                                                     cellDimension))
# Number of vertices
numberOfVertices = int(inFile.readline())
print('%d vertices.' % numberOfVertices)
# Vertices
vertices = [inFile.readline() for _i in range(numberOfVertices)]

# Number of cells.
numberOfCells = int(inFile.readline())

outFile.write('%d %d %d %d\n' % (spaceDimension, cellDimension,
                                 numberOfVertices, numberOfCells))
# The vertices.
for line in vertices:
    outFile.write(line)

#
# The indexed cells.
#
print "%d cells." % numberOfCells
print "Converting C indices to fortran indices..."
for n in range(numberOfCells):
    numberStrings = inFile.readline().split()
    format = "%d" + " %d" * (len(numberStrings) - 1) + "\n"
    outFile.write(format %
                  tuple(map(lambda x: x+1, map(int, numberStrings))))
print "Done."
