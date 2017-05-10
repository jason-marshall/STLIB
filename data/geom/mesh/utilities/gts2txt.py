# gts2txt.py
# Convert a simplicial mesh in GTS format to an ascii file that we can read.
# Usage:
# python gts2txt.py in out

import sys, string

if len(sys.argv) != 3:
    print "Usage:"
    print "python gts2txt.py in out\n"
    print "This program converts a GTS mesh to a mesh in text format."
    raise "Bad command line arguments.  Exiting..."

# Open the input file.
in_file = open(sys.argv[1], "r")


#
# The number of vertices, edges and faces.
#
number_strings = string.split(in_file.readline())
if len(number_strings) != 3:
    raise "Error reading the number of vertices, edges and faces."
num_vertices = string.atoi(number_strings[0])
print "%d vertices." % num_vertices
num_edges = string.atoi(number_strings[1])
print "%d edges." % num_edges
num_faces = string.atoi(number_strings[2])
print "%d faces." % num_faces


#
# The vertices.
#
print "Reading the vertices..."
vertices = []
for n in range(num_vertices):
    vertices.append(in_file.readline())
print "Done."


#
# The (vertex) indexed edges.
#
print "Reading the edges..."
edges = []
for n in range(num_edges):
    number_strings = string.split(in_file.readline())
    edges.append((string.atoi(number_strings[0]) - 1,
                    string.atoi(number_strings[1]) - 1))
print "Done."


#
# The (edge) indexed faces.
#
print "Reading the faces..."
faces = []
for n in range(num_faces):
    number_strings = string.split(in_file.readline())
    faces.append((string.atoi(number_strings[0]) - 1,
                    string.atoi(number_strings[1]) - 1,
                    string.atoi(number_strings[2]) - 1))
print "Done."

in_file.close()




#
# Open the output file.
#
out_file = open(sys.argv[2], "w")


#
# The space dimension and the simplex dimension.
#
out_file.write("3 2\n")


#
# The vertices.
#
out_file.write("%d\n" % num_vertices)
for line in vertices:
    out_file.write(line)


#
# The faces.
#
out_file.write("%d\n" % num_faces)
for face in faces:
    a = edges[ face[0] ][ 0 ]
    b = edges[ face[0] ][ 1 ]
    c = edges[ face[1] ][ 0 ]
    d = edges[ face[1] ][ 1 ]
    if a == c or b == c:
        c = d
    out_file.write("%d %d %d\n" % (a,  b, c))


#
# Close the output file.
#
out_file.close()
