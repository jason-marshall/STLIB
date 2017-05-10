# QuadMeshF2C.py
# Convert a quadrilateral mesh with fortran indices to one with C indices.
# i.e. subtract 1 from each index so the lowest index is 0.
# Usage:
# python QuadMeshF2C.py fortranMesh cMesh

import sys, string

if len(sys.argv) != 3:
    print "Usage:"
    print "python QuadMeshF2C.py fortranMesh cMesh\n"
    print "This program converts a mesh with fortran indices (starting at 1)"
    print "into a mesh with C indices (starting at 0)."
    raise "Bad command line arguments.  Exiting..."

# Open the files.
inFile = open(sys.argv[1], "r")
outFile = open(sys.argv[2], "w")

#
# The space dimension.
#
spaceDimension = string.atoi(inFile.readline())
print "space dimension = %d." % spaceDimension
outFile.write("%d\n" % spaceDimension)

#
# The vertices.
#
numVertices = string.atoi(inFile.readline())
print "%d vertices." % numVertices
outFile.write("%d\n" % numVertices)

print "Reading and writing the vertices..."
for n in range(numVertices):
    outFile.write(inFile.readline())
print "Done."

#
# The indexed simplices.
#
numSimplices = string.atoi(inFile.readline())
print "%d simplices." % numSimplices
outFile.write("%d\n" % numSimplices)

print "Converting fortran indices to C indices..."
format = "%d %d %d %d\n"
for n in range(numSimplices):
    numberStrings = string.split(inFile.readline())
    outFile.write(format %
                    tuple(map(lambda x: x-1,
                                map(string.atoi, numberStrings))))
print "Done."

inFile.close()
outFile.close()

