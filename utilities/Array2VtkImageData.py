# Array2VtkImageData.py
"""
Convert a 2-D or 3-D array to a vtk image data file.
Usage:
python Array2VtkImageData.py in out
"""

import sys, string

if len(sys.argv) != 3:
    print "Usage:"
    print "python Array2VtkImageData.py in out\n"
    print "This program converts a 2-D or 3-D array to a vtk image data file."
    raise "Bad command line arguments.  Exiting..."

# Open the input file.
inFile = open(sys.argv[1], "r")

#
# Index ranges.
#
extents = map(int, inFile.readline().split())
dimension = len(extents)
assert dimension == 2 or dimension == 3
# Bases.
assert len(inFile.readline().split()) == dimension
# Storage order.
assert len(inFile.readline().split()) == dimension

if dimension == 2:
    extents.append(1)

#
# Write the image data.
#
vtkExtents = (0, extents[0] - 1, 0, extents[1] - 1, 0, extents[2] - 1)
outFile = open(sys.argv[2], "w")
outFile.write('<?xml version="1.0"?>\n')
outFile.write('<VTKFile type="ImageData">\n')
outFile.write('<ImageData WholeExtent="%d %d %d %d %d %d" ' % vtkExtents)
outFile.write('Origin="0 0 0" Spacing="1 1 1">\n')
outFile.write('<Piece Extent="%d %d %d %d %d %d">\n' % vtkExtents)
outFile.write('<PointData Scalars="Distance">\n')

outFile.write('<DataArray type="Float32" format="ascii" Name="Distance">\n')
for line in inFile:
    outFile.write(line)
outFile.write('</DataArray>\n')

outFile.write('</PointData>\n')
outFile.write('</Piece>\n')
outFile.write('</ImageData>\n')
outFile.write('</VTKFile>\n')

#
# Close the files.
#
inFile.close()
outFile.close()
