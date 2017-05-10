# vect2.py
# Convert a triangle mesh to a OOGL VECT.
# Usage:
# python vect.py mesh.txt mesh.vect

# CONTINUE: Fix for new format.

import sys, string

if len(sys.argv) != 3:
    raise "Bad command line arguments.  Exiting..."

# Open the files.
in_file = open(sys.argv[1], "r")
out_file = open(sys.argv[2], "w")

# Read the number of vertices and triangles.
number_strings = string.split(in_file.readline())
num_vertices = string.atoi(number_strings[0])
num_triangles = string.atoi(number_strings[1])

# Read the vertices.
vertices = []
for n in range(num_vertices):
    number_strings = string.split(in_file.readline())
    vertices.append(tuple(map(string.atof, number_strings)))

# Read the triangles.
triangles = []
for n in range(num_triangles):
    number_strings = string.split(in_file.readline())
    triangles.append(tuple(map(string.atoi, number_strings)))

in_file.close()

triangle_lines = [(0,1),(1,2),(2,0)]

out_file.write("VECT\n")
# Number of lines, number of vertices, number of colors.
out_file.write("%d %d 1\n" % (3 * num_triangles, 6 * num_triangles))
# Each line is composed of 2 vertices.
for n in range(3 * num_triangles):
    out_file.write("2\n")
# The first line has the first color.
out_file.write("\n1\n")
# All the other lines borrow this color.
for n in range(3 * num_triangles - 1):
    out_file.write("0\n")
out_file.write("\n")
# Write each line.
for tri in triangles:
    for pair in triangle_lines:
        out_file.write("%g %g 0 %g %g 0\n" % (vertices[tri[pair[0]]][0],
                                                vertices[tri[pair[0]]][1],
                                                vertices[tri[pair[1]]][0],
                                                vertices[tri[pair[1]]][1]))

# Write the color
out_file.write("\n0 0 0 1\n")

out_file.close()
