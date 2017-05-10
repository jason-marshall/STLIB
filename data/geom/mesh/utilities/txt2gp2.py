# txt2gp2.py
# Convert a triangle mesh to a gnuplot data file.
# Usage:
# python txt2gp2.py mesh.txt mesh.dat

import sys, string

if len(sys.argv) != 3:
    print "Usage:"
    print "txt2gp2.py mesh.txt mesh.dat\n"
    print "This program converts a triangle mesh to a gnuplot data file."
    raise "Bad command line arguments.  Exiting..."

#
# Open the files.
#
in_file = open(sys.argv[1], "r")
out_file = open(sys.argv[2], "w")

#
# The space dimension and the simplex dimension.
#
number_strings = string.split(in_file.readline())
space_dimension = string.atoi(number_strings[0])
simplex_dimension = string.atoi(number_strings[1])
print "space dimension = %d, simplex dimension = %d" % (space_dimension, 
                                                         simplex_dimension)
if space_dimension != 2:
    raise "Sorry, the space dimension must be 2."
if simplex_dimension != 2:
    raise "Sorry, the simplex dimension must be 2."

#
# Read the vertices.
#
num_vertices = string.atoi(in_file.readline())
print "Reading the %d vertices..." % num_vertices
vertices = []
for n in range(num_vertices):
    number_strings = string.split(in_file.readline())
    vertices.append(tuple(map(string.atof, number_strings)))
print "Done."

#
# Read the triangles.
#
num_triangles = string.atoi(in_file.readline())
print "Reading the %d triangles..." % num_triangles
triangles = []
for n in range(num_triangles):
    number_strings = string.split(in_file.readline())
    triangles.append(tuple(map(string.atoi, number_strings)))
print "Done."

in_file.close()

#
# Write the gnuplot file.
#
print "Writing the gnuplot file..."
triangle_lines = [(0,1),(1,2),(2,0)]
# Put the file name as a comment in the data file.
out_file.write("# " + sys.argv[2] + "\n")
# Write each line.
for tri in triangles:
    for pair in triangle_lines:
        out_file.write("%g %g\n%g %g\n\n" % (vertices[tri[pair[0]]][0],
                                       	       vertices[tri[pair[0]]][1],
                                               vertices[tri[pair[1]]][0],
                                               vertices[tri[pair[1]]][1]))
out_file.close()
print "Done."
