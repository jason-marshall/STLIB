# txt2gp21.py
# Convert a 1-D mesh in 2-D space into a gnuplot data file.
# Usage:
# python txt2gp21.py mesh.txt mesh.dat

import sys, string

if len(sys.argv) != 3:
    print "Usage:"
    print "txt2gp2.py mesh.txt mesh.dat\n"
    print "This program converts a 1-D mesh in 2-D to a gnuplot data file."
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
if simplex_dimension != 1:
    raise "Sorry, the simplex dimension must be 1."

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
# Read the line segments.
#
num_simplices = string.atoi(in_file.readline())
print "Reading the %d simplices..." % num_simplices
simplices = []
for n in range(num_simplices):
    number_strings = string.split(in_file.readline())
    simplices.append(tuple(map(string.atoi, number_strings)))
print "Done."

in_file.close()

#
# Write the gnuplot file.
#
print "Writing the gnuplot file..."
#triangle_lines = [(0,1),(1,2),(2,0)]
# Put the file name as a comment in the data file.
out_file.write("# " + sys.argv[2] + "\n")
# Write each line.
for s in simplices:
    out_file.write("%g %g\n%g %g\n\n" % (vertices[s[0]][0],
                                           vertices[s[0]][1],
                                           vertices[s[1]][0],
                                           vertices[s[1]][1]))
out_file.close()
print "Done."
