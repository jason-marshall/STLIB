# txt2gp3.py
# Convert a tetrahedral mesh to a gnuplot data file.
# Usage:
# python txt2gp3.py mesh.txt mesh.dat

import sys, string

if len(sys.argv) != 3:
    print "Usage:"
    print "txt2gp3.py mesh.txt mesh.dat\n"
    print "This program converts a tetrahedral mesh to a gnuplot data file."
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
if space_dimension != 3:
    raise "Sorry, the space dimension must be 3."
if simplex_dimension != 3:
    raise "Sorry, the simplex dimension must be 3."


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
# Read the tets.
#
num_tets = string.atoi(in_file.readline())
print "Reading the %d tetrahedra..." % num_tets
tets = []
for n in range(num_tets):
    number_strings = string.split(in_file.readline())
    tets.append(tuple(map(string.atoi, number_strings)))
print "Done."

in_file.close()

#
# Write the gnuplot file.
#
print "Writing the gnuplot file..."
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
print "Done."
