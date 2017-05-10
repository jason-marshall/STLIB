# cone32.py
# Make a triangle mesh for the surface of a cone with unit radius and height.
# Usage:
# python cone32.py numPoints outputMesh
# - numPoints is the number of points around the base of the cone.

import sys
from math import *

if len(sys.argv) != 3:
    print "Usage:"
    print "python cone32.py numPoints outputMesh"
    print "- numPoints is the number of points around the base of the cone."
    raise "Wrong number of command line arguments.  Exiting..."

# Get the number of points.
numPoints = eval(sys.argv[1])
if numPoints < 3:
    print "You specified " << numPoints << "points."
    raise "Error: Bad number of points.  Exiting..."

# Write the mesh.
print "Writing the output mesh...."
outFile = open(sys.argv[2], "w")
# Dimensions.
outFile.write("3 2\n")
# Number of vertices.
outFile.write("%d\n" % (numPoints + 2))
# First two vertices.
outFile.write("0 0 0\n")
outFile.write("0 0 1\n")
# The rest of the vertices.
for n in range(0,numPoints):
    outFile.write("%.16g %.16g 0\n" % ((cos(n * 2.0 * pi / numPoints),
                                        sin(n * 2.0 * pi / numPoints))))
# Number of simplices.    
outFile.write("%d\n" % (2 * numPoints))
for n in range(0, numPoints):
    outFile.write("0 %d %d\n" % ((n + 1) % numPoints + 2, n + 2))
    outFile.write("1 %d %d\n" % (n + 2, (n + 1) % numPoints + 2))
outFile.close()
print "Done."
