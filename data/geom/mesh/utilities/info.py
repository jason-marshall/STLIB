# info.py
# Print information about a simplicial mesh.
# Usage:
# python info.py mesh

import sys, string

if len(sys.argv) != 2:
    print "Usage:"
    print "python info.py mesh\n"
    print "This program prints information about a mesh.  It reads the mesh"
    print "in ascii format."
    raise "Bad command line arguments.  Exiting..."

# Open the input file.
in_file = open(sys.argv[1], "r")

#
# The space dimension and the simplex dimension.
#
number_strings = string.split(in_file.readline())
space_dimension = string.atoi(number_strings[0])
simplex_dimension = string.atoi(number_strings[1])
print "space dimension = %d, simplex dimension = %d" % (space_dimension, 
                                                         simplex_dimension)
#
# The number of vertices.
#
num_vertices = string.atoi(in_file.readline())
print "%d vertices." % num_vertices

#
# Read the vertices.
#
lower = [0] * space_dimension
upper = [0] * space_dimension
if num_vertices >= 1:
    print "Reading the vertices..."
    # Initialize the bounding box with vertex 0.
    number_strings = string.split(in_file.readline())
    for n in range(space_dimension):
        lower[n] = upper[n] = string.atof(number_strings[n])

    # Read the rest of the vertices.
    for n in range(1, num_vertices):
        number_strings = string.split(in_file.readline())
        for n in range(space_dimension):
            v = string.atof(number_strings[n])
            if v < lower[n]:
                lower[n] = v
            elif v > upper[n]:
                upper[n] = v
    print "Done."
    print "Bounding box around the vertices:"
    bbox = "[%g .. %g]" % (lower[0], upper[0])
    for n in range(1, space_dimension):
        bbox += " x [%g .. %g]" % (lower[n], upper[n])
    print bbox

#
# The number of elements.
#
num_elements = string.atoi(in_file.readline())
print "%d elements." % num_elements

#
# Read the indexed elements.
#
if num_elements >= 1:
    print "Reading the indexed elements..."
    indices = []
    for n in range(num_elements):
        number_strings = string.split(in_file.readline())
        for ns in number_strings:
            indices.append(string.atoi(ns))
    print "Done."
    print "There are %d nodes per element." % len(number_strings)
    print "Min index = %d.  Max index = %d." % (min(indices), max(indices))
in_file.close()
