"""nodesToVtk.py
Convert ascii nodes data to a VTK XML unstructured mesh file.

Usage:
python nodesToVtk.py in.txt out.vtu"""

import sys, string

if len(sys.argv) != 3:
    print "Usage:"
    print "python nodesToVtk.py in.txt out.vtu\n"
    print "This program converts ascii nodes data to a VTK XML unstructured mesh file."
    raise "Bad command line arguments.  Exiting..."

#
# Read the node data.
#
positions = []
for line in open(sys.argv[1], "r").readlines():
    numberStrings = line.split()
    if len(numberStrings) == 3:
        positions.append(tuple(map(float, numberStrings)))
    else:
        print "Ignoring line:\n%s" % line

#
# Write the VTK file.
#
outFile = open(sys.argv[2], "w")
outFile.write("<?xml version=\"1.0\"?>\n")
outFile.write("<VTKFile type=\"UnstructuredGrid\">\n")
outFile.write("<UnstructuredGrid>\n")
outFile.write('<Piece NumberOfPoints="' + str(len(positions))
              + '" NumberOfCells="' + str(len(positions)) + '">\n')

outFile.write("<PointData>\n")
outFile.write("</PointData>\n")

outFile.write("<CellData>\n")
outFile.write("</CellData>\n")

outFile.write("<Points>\n")
outFile.write("<DataArray type=\"Float32\" NumberOfComponents=\"3\">\n")
for p in positions:
    outFile.write("%f %f %f\n" % p)
outFile.write("</DataArray>\n")
outFile.write("</Points>\n")

outFile.write("<Cells>\n")
outFile.write('<DataArray type="Int32" Name="connectivity">\n')
for i in range(len(positions)):
    outFile.write("%d\n" % i)
outFile.write("</DataArray>\n")
outFile.write('<DataArray type="Int32" Name="offsets">\n')
for i in range(1, len(positions) + 1):
    outFile.write("%d\n" % i)
outFile.write("</DataArray>\n")
outFile.write('<DataArray type="UInt8" Name="types">\n')
for i in range(len(positions)):
    outFile.write("1\n")
outFile.write("</DataArray>\n")
outFile.write("</Cells>\n")

outFile.write("</Piece>\n")
outFile.write("</UnstructuredGrid>\n")
outFile.write("</VTKFile>\n")
outFile.close()
