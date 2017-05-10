# c2f.py
# Convert a simplicial mesh with C indices to one with fortran indices,
# i.e. add 1 to each index so the lowest index is 1.
# Usage:
# python c2f.py cMesh fortranMesh

import sys

if len(sys.argv) != 3:
    print("Usage:")
    print("python c2f.py cMesh fortranMesh\n")
    print("This program converts a mesh with C indices (starting at 0)")
    print("into a mesh with fortran indices (starting at 1).")
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
assert len(numberStrings) == 2
spaceDimension = int(numberStrings[0])
simplexDimension = int(numberStrings[1])
print("space dimension = %d, simplex dimension = %d" % (spaceDimension, 
                                                        simplexDimension))
outFile.write("%d %d\n" % (spaceDimension, simplexDimension))

#
# The vertices.
#
numVertices = int(inFile.readline())
print("%d vertices." % numVertices)
outFile.write("%d\n" % numVertices)

print("Reading and writing the vertices...")
for n in range(numVertices):
    outFile.write(inFile.readline())
print("Done.")

#
# The indexed simplices.
#
numSimplices = int(inFile.readline())
print("%d simplices." % numSimplices)
outFile.write("%d\n" % numSimplices)

print("Converting C indices to fortran indices...")
for n in range(numSimplices):
    numberStrings = inFile.readline().split()
    outFile.write(' '.join(map(str, map(lambda x: x+1,
                                        map(int, numberStrings)))) + '\n')
print("Done.")
