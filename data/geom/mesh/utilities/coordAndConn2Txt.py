# coordAndConn2Txt.py
# Convert an array of coordinates and an array of connectivities to an
# ascii file that we can read.
# Usage:
# python coordAndConn2Txt.py coordinates connectivities mesh

import sys, string

if len(sys.argv) != 4:
    print "Usage:"
    print "python coordAndConn2Txt.py coordinates connectivities mesh\n"
    print "Convert an array of coordinates and an array of connectivities to"
    print "an ascii file that we can read."
    raise "Bad command line arguments.  Exiting..."

#
# Read the coordinates file.
#
coordinatesFile = open(sys.argv[1], "r")
coordinateLines = coordinatesFile.readlines()
coordinatesFile.close()
if len(coordinateLines) == 0:
    raise "Error: The coordinates are empty.  Exiting..."
spaceDimension = len(string.split(coordinateLines[0]))
if spaceDimension == 0:
    raise "Error in read the coordinates.  Exiting..."
print "The space dimension is %d." % spaceDimension
coordinates = []
for line in coordinateLines:
    if len(string.split(line)) == spaceDimension:
        coordinates.append(line)
coordinatesSize = len(coordinates)
print "There are %d coordinates." % coordinatesSize

#
# Read the connectivities file.
#
connectivitiesFile = open(sys.argv[2], "r")
connectivityLines = connectivitiesFile.readlines()
connectivitiesFile.close()
if len(connectivityLines) == 0:
    raise "Error: The connectivities are empty.  Exiting..."
simplexDimension = len(string.split(connectivityLines[0])) - 1
if simplexDimension < 0:
    raise "Error in read the connectivities.  Exiting..."
print "The simplex dimension is %d." % simplexDimension
connectivities = []
for line in connectivityLines:
    if len(string.split(line)) == simplexDimension + 1:
        connectivities.append(line)
connectivitiesSize = len(connectivities)
print "There are %d elements." % connectivitiesSize

#
# Write the mesh file.
#
# Open the output file.
meshFile = open(sys.argv[3], "w")
# The space dimension and the simplex dimension.
meshFile.write("%d %d\n" % (spaceDimension, simplexDimension))
# The vertices.
meshFile.write("%d\n" % coordinatesSize)
for line in coordinates:
    meshFile.write(line)
# The elements.
meshFile.write("%d\n" % connectivitiesSize)
for line in connectivities:
    meshFile.write(line)
# Close the output file.
meshFile.close()
