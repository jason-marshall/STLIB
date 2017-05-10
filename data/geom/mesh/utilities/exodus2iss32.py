# exodus2iss32.py
# Convert a simplicial mesh in Exodus II ascii format to a 3-2 indexed simplex
# set ascii file.
# Usage:
# python exodus2iss32.py in out
#
# Use the ncdump program to convert a binary Exodus II file to an ascii file.

import sys, string, re

if len(sys.argv) != 3:
    print "Usage:"
    print "python exodus2iss32.py in out\n"
    print "This program converts an exodus II mesh to a mesh in text format."
    raise "Bad command line arguments.  Exiting..."


#
# Read the ascii Exodus file into a string.
#

# Open the input file.
inputFile = open(sys.argv[1], "r")
# Read the input file.
input = inputFile.read()
# Close the input file.
inputFile.close()
print 'The input file has %d characters' % len(input)


#
# Get the vertices.
#

# Get the number of vertices.
m = re.search(r'num_nodes\s*\=\s*(\d+)\s*\;', input)
numberOfVertices = eval(m.group(1))
print '%d vertices.' % numberOfVertices

print "Reading the vertices..."
vertices = []
# Get The coordinates string.  It starts with 'coord' and ends with 
# a semicolon.
# Note: This is simpler, but triggers a "maximum recursion limit exceeded"
# error in python 2.1
#coordinatesRegEx = re.compile(r'coord\s*\=.*?\;', re.S)
coordinatesRegEx = re.compile(r'coord\s*\=[\-\+0-9\.\,eE\s\n]*\;')
m = coordinatesRegEx.search(input)
s = m.group()
# Get the number strings.

floatingPointNumberRegex = re.compile(r'\-?\d+[\-\w\.]*')
vertexStrings = floatingPointNumberRegex.findall(s)
if numberOfVertices * 3 != len(vertexStrings):
  raise('Error reading the vertices')
# All the x coordinates come first, followed by y and z.
# We leave the vertices as strings.  Then we don't have to worry about
# changing the precision.
for n in range(numberOfVertices):
  vertices.append('%s %s %s' % (vertexStrings[n],
                                vertexStrings[numberOfVertices + n],
                                vertexStrings[2 * numberOfVertices + n]))
print "Done."


#
# The faces.
#

# Get the number of faces.
m = re.search(r'num_elem\s*\=\s*(\d+)\s*\;', input)
numberOfFaces = eval(m.group(1))
print '%d faces.' % numberOfFaces

print "Reading the faces..."
faces = []
# Get each connectivities group.
indexRegEx = re.compile(r'\d+')
# Note: This is simpler, but triggers a "maximum recursion limit exceeded"
# error in python 2.1
#connectivitiesRegEx = re.compile(r'connect\d+\s*\=.*?\;', re.S)
connectivitiesRegEx = re.compile(r'connect\d+\s*\=[0-9\,\s\n]*\;')
faceStrings = connectivitiesRegEx.findall(input)
for s in faceStrings:
    # Get rid of the variable name.
    s = re.sub(r'connect\d+\s*\=', '', s)
    numberStrings = indexRegEx.findall(s)
    if len(numberStrings) % 3 != 0:
        raise('Error in reading the faces.')
    for n in range(len(numberStrings) / 3):
        faces.append((eval(numberStrings[3*n]) - 1,
                      eval(numberStrings[3*n+1]) - 1,
                      eval(numberStrings[3*n+2]) - 1))
print "Done."


#
# Open the output file.
#
outputFile = open(sys.argv[2], "w")


#
# The space dimension and the simplex dimension.
#
outputFile.write("3 2\n")


#
# The vertices.
#
outputFile.write("%d\n" % numberOfVertices)
for v in vertices:
   outputFile.write("%s\n" % v)


#
# The faces.
#
outputFile.write("%d\n" % numberOfFaces)
for f in faces:
    outputFile.write("%d %d %d\n" % f)


#
# Close the output file.
#
outputFile.close()
