# vertices_map.py
# Apply a mapping to the vertices of a mesh.
# Usage:
# python vertices_map.py function [subset] in out

import sys, string
from math import *

if not (len(sys.argv) == 4 or len(sys.argv) == 5):
    print("Usage:")
    print("python vertices_map.py function [subset] in out\n")
    print("Bad command line arguments.  Exiting...")
    sys.exit(1)

#
# Open the input files.
#
arg_index = 1
function_file = open(sys.argv[arg_index], "r")
arg_index += 1

subset_file = None
if len(sys.argv) == 5:
    subset_file = open(sys.argv[arg_index], "r")
    arg_index += 1

in_file = open(sys.argv[arg_index], "r")
arg_index += 1

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
    pt = tuple(map(string.atof, number_strings))
    vertices.append(pt)
print "Done."

#
# Read the simplices.
#
num_simplices = string.atoi(in_file.readline())
print "Reading the %d simplices...." % num_simplices
simplices = []
for n in range(num_simplices):
    simplices.append(in_file.readline())
print "Done."
in_file.close()

#
# Get the subset of the vertices.
#
if subset_file:
    subset = []
    subset_size = string.atoi(subset_file.readline())
    for n in range(subset_size):
        subset.append(string.atoi(subset_file.readline()))
    subset_file.close()
else:
    subset = range(len(vertices))
print "Done."


#
# Apply a function to the vertices.
#
print "Applying the function..."
if space_dimension == 1:
    exec("def a(x): return " + function_file.readline()[0:-1])
    for n in subset:
        x = vertices[n][0]
        vertices[n] = (a(x))
elif space_dimension == 2:
    exec("def a(x,y): return " + function_file.readline()[0:-1])
    exec("def b(x,y): return " + function_file.readline()[0:-1])
    for n in subset:
        x = vertices[n][0]
        y = vertices[n][1]
        vertices[n] = (a(x, y), b(x, y))
elif space_dimension == 3:
    exec("def a(x,y,z): return " + function_file.readline()[0:-1])
    exec("def b(x,y,z): return " + function_file.readline()[0:-1])
    exec("def c(x,y,z): return " + function_file.readline()[0:-1])
    for n in subset:
        x = vertices[n][0]
        y = vertices[n][1]
        z = vertices[n][2]
        vertices[n] = (a(x,y,z), b(x,y,z), c(x,y,z))
else:
    raise "Bad space dimension."

function_file.close()
print "Done."

#
# Write the mesh.
#
print "Writing the output mesh...."
out_file = open(sys.argv[arg_index], "w")
arg_index += 1
out_file.write("%d %d\n" % (space_dimension, simplex_dimension))
out_file.write("%d\n" % num_vertices)
format_string = "%.16g" + " %.16g" * (space_dimension - 1) + "\n"
for vertex in vertices:
    out_file.write(format_string % vertex)
out_file.write("%d\n" % num_simplices)
for line in simplices:
    out_file.write(line)
out_file.close()
print "Done."
