# vect.py
# Convert a tet mesh to a OOGL VECT.
# Usage:
# python vect.py mesh.txt mesh.vect

# CONTINUE: Fix for new format.

import sys, string

if len(sys.argv) != 3:
    raise "Bad command line arguments.  Exiting..."

# Open the files.
in_file = open(sys.argv[1], "r")
out_file = open(sys.argv[2], "w")

# Read the number of vertices and tets.
number_strings = string.split(in_file.readline())
num_vertices = string.atoi(number_strings[0])
num_tets = string.atoi(number_strings[1])

# Read the vertices.
vertices = []
for n in range(num_vertices):
    number_strings = string.split(in_file.readline())
    vertices.append(tuple(map(string.atof, number_strings)))

# Read the tets.
tets = []
for n in range(num_tets):
    number_strings = string.split(in_file.readline())
    tets.append(tuple(map(string.atoi, number_strings)))

in_file.close()

tet_lines = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]

out_file.write("VECT\n")
# Number of lines, number of vertices, number of colors.
out_file.write("%d %d 1\n" % (6 * num_tets, 12 * num_tets))
# Each line is composed of 2 vertices.
for n in range(6 * num_tets):
    out_file.write("2\n")
# The first line has the first color.
out_file.write("\n1\n")
# All the other lines borrow this color.
for n in range(6 * num_tets - 1):
    out_file.write("0\n")
out_file.write("\n")
# Write each line.
for tet in tets:
    for pair in tet_lines:
        out_file.write("%g %g %g %g %g %g\n" % (vertices[tet[pair[0]]][0],
                                                  vertices[tet[pair[0]]][1],
                                                  vertices[tet[pair[0]]][2],
                                                  vertices[tet[pair[1]]][0],
                                                  vertices[tet[pair[1]]][1],
                                                  vertices[tet[pair[1]]][2]))

# Write the color
out_file.write("\n0 0 0 1\n")

out_file.close()
