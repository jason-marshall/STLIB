# adlib2stlib.py
# Convert an adlib mesh to an stlib mesh.
# python adlib2stlib.py x.txt y.txt

import sys, string

if len(sys.argv) != 3:
    print "Usage:"
    print "python adlib2stlib.py x.txt y.txt\n"
    print "Bad command line arguments.  Exiting..."
    sys.exit()

# Open the files.
assert sys.argv[1] != sys.argv[2]
inFile = open(sys.argv[1], "r")
outFile = open(sys.argv[2], "w")

#
# Space dimension, cell dimension, number of vertices, number of cells.
#
numberStrings = inFile.readline().split()
assert len(numberStrings) == 4
spaceDimension = int(numberStrings[0])
cellDimension = int(numberStrings[1])
numberOfVertices = int(numberStrings[2])
numberOfCells = int(numberStrings[3])
print('space dimension = %d, cell dimension = %d' % (spaceDimension,
                                                     cellDimension))
outFile.write('%d %d\n' % (spaceDimension, cellDimension))
# The vertices.
print('%d vertices.' % numberOfVertices)
outFile.write('%d\n' % numberOfVertices)

print "Reading and writing the vertices..."
for n in range(numberOfVertices):
    outFile.write(inFile.readline())
print "Done."

#
# The indexed cells.
#
print "%d cells." % numberOfCells
outFile.write("%d\n" % numberOfCells)

print "Converting fortran indices to C indices..."
for n in range(numberOfCells):
    numberStrings = inFile.readline().split()
    format = "%d" + " %d" * (len(numberStrings) - 1) + "\n"
    outFile.write(format %
                  tuple(map(lambda x: x-1, map(int, numberStrings))))
print "Done."
