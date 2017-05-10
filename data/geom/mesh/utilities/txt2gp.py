# txt2gp3.py
# Convert a tetrahedral mesh to a gnuplot data file.
# Usage:
# python txt2gp3.py mesh.txt mesh.dat

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

# Put the file name as a comment in the data file.
out_file.write("# " + sys.argv[2] + "\n")
# Write each line.
for tet in tets:
    for pair in tet_lines:
        out_file.write("%g %g %g\n%g %g %g\n\n" % 
	(vertices[tet[pair[0]]][0],
        	                                  vertices[tet[pair[0]]][1],
                                                  vertices[tet[pair[0]]][2],
                                                  vertices[tet[pair[1]]][0],
                                                  vertices[tet[pair[1]]][1],
                                                  vertices[tet[pair[1]]][2]))

out_file.close()
