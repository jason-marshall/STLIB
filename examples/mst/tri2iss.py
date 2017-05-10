# tri2iss.py
# Naively convert a set of triangles to an indexed simplex set.  "Naively",
# because, we don't merge duplicate vertices.
# Usage:
# python tri2iss.py in out

import sys, string

if len(sys.argv) != 3:
  print "Usage:"
  print "python tri2iss.py in out\n"
  print "Naively convert a set of triangles to an indexed simplex set."
  raise "Bad command line arguments.  Exiting..."


# Open the input file.
in_file = open(sys.argv[1], "r")

# The number of triangles.
number_strings = string.split(in_file.readline())
if len(number_strings) != 1:
  raise "Error reading the number of triangles."
num_triangles = int(number_strings[0])
print "%d triangles." % num_triangles


# The vertices.
print "Reading the vertices..."
vertex_strings = in_file.readlines()
if (len(vertex_strings) < 3*num_triangles):
  raise "Not enough vertices in the input file."
# Ignore anything after the vertices.
vertex_strings = vertex_strings[0:3*num_triangles]
print "Done."

# Close the input file.
in_file.close()


# Open the output file.
out_file = open(sys.argv[2], "w")

# The space dimension and the simplex dimension.
out_file.write("3 2\n")

# The vertices.
out_file.write("%d\n" % (3*num_triangles))
for line in vertex_strings:
  out_file.write("%s" % line)

# The indexed faces.
out_file.write("%d\n" % num_triangles)
for n in range(num_triangles):
  out_file.write("%d %d %d\n" % (3*n, 3*n+1, 3*n+2))

# Close the output file.
out_file.close()
