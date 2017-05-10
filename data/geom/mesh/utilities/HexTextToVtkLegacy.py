# HexTextToVtkLegacy.py
# Convert a hexahedral mesh from text format to VTK legacy format.
# Usage:
# python HexTextToVtkLegacy.py input output

import sys, string

if len(sys.argv) != 3:
    print "Usage:"
    print "python HexTextToVtkLegacy.py input output\n"
    print "Convert a hexahedral mesh from text format to VTK legacy format."
    raise "Bad command line arguments.  Exiting..."

# Open the files.
inFile = open(sys.argv[1], "r")
outFile = open(sys.argv[2], "w")

#
# The space dimension and the cell dimension.
#
numberStrings = string.split(inFile.readline())
spaceDimension = int(numberStrings[0])
cellDimension = int(numberStrings[1])
if spaceDimension != 3:
    raise "Bad space dimension."
if cellDimension != 3:
    raise "Bad cell dimension."
print "space dimension = %d, cell dimension = %d" % (spaceDimension, 
                                                         cellDimension)
outFile.write("# vtk DataFile Version 2.0\n")
outFile.write("Hexahedral Mesh\n")
outFile.write("ASCII\n\n")

#
# The vertices.
#
numberOfVertices = int(inFile.readline())
print "%d vertices." % numberOfVertices
outFile.write("DATASET UNSTRUCTURED_GRID\n")
outFile.write("POINTS %d float\n" % numberOfVertices)

print "Reading and writing the vertices..."
for n in range(numberOfVertices):
    outFile.write(inFile.readline())
outFile.write("\n")
print "Done."

#
# The indexed cells.
#
numberOfCells = int(inFile.readline())
print "%d cells." % numberOfCells

print "Reading and writing the cells..."
outFile.write("CELLS %d %d\n" % (numberOfCells, 8 * numberOfCells))
for n in range(numberOfCells):
    outFile.write("8 " + inFile.readline())
print "Done."

print "Writing the cell types..."
outFile.write("CELL_TYPES %d\n" % numberOfCells)
for n in range(numberOfCells):
    outFile.write("12\n")
print "Done."

inFile.close()
outFile.close()
