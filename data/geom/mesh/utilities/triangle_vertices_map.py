# triangle_vertices_map.py
# Apply a mapping to the vertices of a triangle mesh.
# Usage:
# python triangle_vertices_map.py in out function

import sys, string

if len(sys.argv) != 4:
    print "Usage:"
    print "python triangle_vertices_map.py in out function\n"
    raise "Bad command line arguments.  Exiting..."

#
# Open the files.
#
in_file = open(sys.argv[1], "r")
out_file = open(sys.argv[2], "w")
function_file = open(sys.argv[3], "r")

#
# The space dimension and the simplex dimension.
#
number_strings = string.split(in_file.readline())
space_dimension = string.atoi(number_strings[0])
simplex_dimension = string.atoi(number_strings[1])
print "space dimension = %d, simplex dimension = %d" % (space_dimension, 
                                                         simplex_dimension)

#
# Read the vertices.
#
num_vertices = string.atoi(in_file.readline())
print "Reading the %d vertices..." % num_vertices
vertices = []
for n in range(num_vertices):
    number_strings = string.split(in_file.readline())
    pt = (string.atof(number_strings[0]),
           string.atof(number_strings[1]))
    vertices.append(pt)
print "Done."

#
# Read the triangles.
#
num_triangles = string.atoi(in_file.readline())
print "Reading the %d triangles...." % num_triangles
triangles = []
for n in range(num_triangles):
    number_strings = string.split(in_file.readline())
    triangles.append((string.atoi(number_strings[0]),
                        string.atoi(number_strings[1]),
                        string.atoi(number_strings[2])))
print "Done."
in_file.close()

#
# Apply a function to the vertices.
#

print "Applying the function..."
exec("def a(x,y): return " + function_file.readline()[0:-1])
exec("def b(x,y): return " + function_file.readline()[0:-1])

for n in range(len(vertices)):
    x = vertices[n][0]
    y = vertices[n][1]
    vertices[n] = (a(x, y), b(x, y))
function_file.close()
print "Done."

#
# Write the triangle mesh.
#
print "Writing the output mesh...."
out_file.write("%d %d\n" % (space_dimension, simplex_dimension))
out_file.write("%d\n" % num_vertices)
for vertex in vertices:
    out_file.write("%g %g\n" % vertex)
out_file.write("%d\n" % num_triangles)
for triangle in triangles:
    out_file.write("%d %d %d\n" % triangle)
out_file.close()
print "Done."
