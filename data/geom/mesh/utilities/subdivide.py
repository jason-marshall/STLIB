# subdivide.py
# Subdivide a triangle mesh.
# Usage:
# python subdivide.py in out

import sys, string

if len(sys.argv) != 3:
    print "Usage:"
    print "python subdivide.py in out\n"
    raise "Bad command line arguments.  Exiting..."

#
# The space dimension and the simplex dimension.
#
in_file = open(sys.argv[1], "r")
number_strings = string.split(in_file.readline())
space_dimension = string.atoi(number_strings[0])
simplex_dimension = string.atoi(number_strings[1])
print "space dimension = %d, simplex dimension = %d" % (space_dimension, 
                                                         simplex_dimension)
if not (space_dimension == 2 or space_dimension == 3):
    raise "Sorry, the space dimension must be 2 or 3."
if simplex_dimension != 2:
    raise "Sorry, the simplex dimension must be 2."

#
# Read the vertices.
#
num_vertices = string.atoi(in_file.readline())
print "Reading the %d vertices..." % num_vertices
vertices = []
vertex_dict = {}
for n in range(num_vertices):
    number_strings = string.split(in_file.readline())
    pt = tuple(map(string.atof, number_strings))
    vertices.append(pt)
    vertex_dict[pt] = n
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
in_file.close()
print "Done."

#
# Subdivide.
#
new_triangles = []
for i in triangles:
    if space_dimension == 2:
        a = ((vertices[i[0]][0] + vertices[i[1]][0]) / 2,
              (vertices[i[0]][1] + vertices[i[1]][1]) / 2)
        b = ((vertices[i[1]][0] + vertices[i[2]][0]) / 2,
              (vertices[i[1]][1] + vertices[i[2]][1]) / 2)
        c = ((vertices[i[2]][0] + vertices[i[0]][0]) / 2,
              (vertices[i[2]][1] + vertices[i[0]][1]) / 2)
    else:
        a = ((vertices[i[0]][0] + vertices[i[1]][0]) / 2,
              (vertices[i[0]][1] + vertices[i[1]][1]) / 2,
              (vertices[i[0]][2] + vertices[i[1]][2]) / 2)
        b = ((vertices[i[1]][0] + vertices[i[2]][0]) / 2,
              (vertices[i[1]][1] + vertices[i[2]][1]) / 2,
              (vertices[i[1]][2] + vertices[i[2]][2]) / 2)
        c = ((vertices[i[2]][0] + vertices[i[0]][0]) / 2,
              (vertices[i[2]][1] + vertices[i[0]][1]) / 2,
              (vertices[i[2]][2] + vertices[i[0]][2]) / 2)
    for p in [a, b, c]:
        if not vertex_dict.has_key(p):
            vertices.append(p)
            vertex_dict[p] = len(vertices) - 1
    new_triangles.append((i[0], vertex_dict[a], vertex_dict[c]))
    new_triangles.append((vertex_dict[a], i[1], vertex_dict[b]))
    new_triangles.append((vertex_dict[c], vertex_dict[b], i[2]))
    new_triangles.append((vertex_dict[a], vertex_dict[b], vertex_dict[c]))
print "In the subdivided mesh there are %d vertices and %d triangles" % \
      (len(vertices), len(new_triangles))

#
# Write the new triangle mesh.
#
print "Writing the subdivided mesh..."
out_file = open(sys.argv[2], "w")
out_file.write("%d %d\n" % (space_dimension, simplex_dimension))
out_file.write("%d\n" % len(vertices))
if space_dimension == 2:
    format = "%g %g\n"
else:
    format = "%g %g %g\n"
for vertex in vertices:
    out_file.write(format % vertex)
out_file.write("%d\n" % len(new_triangles))
for triangle in new_triangles:
    out_file.write("%d %d %d\n" % triangle)
out_file.close()
