# exodus2iss33.py
# Convert a simplicial mesh in Exodus II ascii format to a 3-3 indexed simplex
# set ascii file.
# Usage:
# python exodus2iss33.py in out
#
# Use the ncdump program to convert a binary Exodus II file to an ascii file.

import sys, string, re

if len(sys.argv) != 3:
    print "Usage:"
    print "python exodus2iss33.py in out\n"
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

# The coordinates may be separated by x, y and z components:
# coordx = ...; coordy = ...; coordz = ...;
# or they may be listed together:
# coord = ...;

floatingPointNumberRegex = re.compile(r'\-?\d+[\-\w\.]*')
# If the coordinates are separated by x, y and z components.
if (re.search(r'coordx', input)):
  # Get The coordinate strings.  They start with 'coord' and end with 
  # a semicolon.

  # This could be done more simply with ., but I got max recursion errors
  # with python 2.1.
  coordinatesRegEx = re.compile(r'coordx\s*\=[\-\+0-9\.\,eE\s\n]*\;')
  m = coordinatesRegEx.search(input)
  xStrings = floatingPointNumberRegex.findall(m.group())

  coordinatesRegEx = re.compile(r'coordy\s*\=[\-\+0-9\.\,eE\s\n]*\;')
  m = coordinatesRegEx.search(input)
  yStrings = floatingPointNumberRegex.findall(m.group())

  coordinatesRegEx = re.compile(r'coordz\s*\=[\-\+0-9\.\,eE\s\n]*\;')
  m = coordinatesRegEx.search(input)
  zStrings = floatingPointNumberRegex.findall(m.group())

  if (len(xStrings) != numberOfVertices or
      len(yStrings) != numberOfVertices or
      len(zStrings) != numberOfVertices):
    raise('Error reading the vertices')
  # We leave the vertices as strings.  Then we don't have to worry about
  # changing the precision.
  for n in range(numberOfVertices):
    vertices.append('%s %s %s' % (xStrings[n], yStrings[n], zStrings[n]))
# Otherwise, the coordinates are listed together.
else:
  # Get The coordinate string.  It starts with 'coord' and ends with 
  # a semicolon.
  coordinatesRegEx = re.compile(r'coord\s*\=[\-\+0-9\.\,eE\s\n]*\;')
  m = coordinatesRegEx.search(input)
  strings = floatingPointNumberRegex.findall(m.group())

  if (len(strings) / 3 != numberOfVertices or
      len(strings) % 3 != 0):
    raise('Error reading the vertices')
  # We leave the vertices as strings.  Then we don't have to worry about
  # changing the precision.
  for n in range(numberOfVertices):
    vertices.append('%s %s %s' % (strings[n], strings[n + numberOfVertices],
                                  strings[n + 2 * numberOfVertices]))

print "Done."


#
# The simplices.
#

# Get the number of simplices.
m = re.search(r'num_elem\s*\=\s*(\d+)\s*\;', input)
numberOfSimplices = eval(m.group(1))
print '%d simplices.' % numberOfSimplices

print "Reading the indexed simplices..."
simplices = []
# Get each connectivities group.
indexRegEx = re.compile(r'\d+')
# Note: This is simpler, but triggers a "maximum recursion limit exceeded"
# error in python 2.1
#connectivitiesRegEx = re.compile(r'connect\d+\s*\=.*?\;', re.S)
connectivitiesRegEx = re.compile(r'connect\d+\s*\=[0-9\,\s\n]*\;')
simplexStrings = connectivitiesRegEx.findall(input)
for s in simplexStrings:
    # Get rid of the variable name.
    s = re.sub(r'connect\d+\s*\=', '', s)
    numberStrings = indexRegEx.findall(s)
    if len(numberStrings) % 4 != 0:
        raise('Error in reading the indexed simplices.')
    for n in range(len(numberStrings) / 4):
        simplices.append((eval(numberStrings[4*n]) - 1,
                          eval(numberStrings[4*n+1]) - 1,
                          eval(numberStrings[4*n+2]) - 1,
                          eval(numberStrings[4*n+3]) - 1))
print "Done."


#
# Open the output file.
#
outputFile = open(sys.argv[2], "w")


#
# The space dimension and the simplex dimension.
#
outputFile.write("3 3\n")


#
# The vertices.
#
outputFile.write("%d\n" % numberOfVertices)
for v in vertices:
   outputFile.write("%s\n" % v)


#
# The indexed simplices.
#
outputFile.write("%d\n" % numberOfSimplices)
for s in simplices:
    outputFile.write("%d %d %d %d\n" % s)


#
# Close the output file.
#
outputFile.close()
