# merge.py
# Merge two meshes.
#
# Usage:
# python merge.py in1 in2 out

import sys

if len(sys.argv) != 4:
    print("Usage:")
    print("python merge.py in1 in2 out\n")
    print("Bad command line arguments.  Exiting...")
    sys.exit(1)

#
# Read the first file.
#

#
# The space dimension and the simplex dimension.
#
file = open(sys.argv[1], "r")
number_strings = file.readline().split()
space_dimension = int(number_strings[0])
simplex_dimension = int(number_strings[1])
print "space dimension = %d, simplex dimension = %d" % (space_dimension, 
                                                         simplex_dimension)

#
# Read the nodes.
#
num_nodes = int(file.readline())
print "Reading the %d node locations from the first file..." % num_nodes
node_strings = []
for n in range(num_nodes):
    node_strings.append(file.readline())
print "Done."

#
# Read the elements.
#
num = int(file.readline())
print "Reading the %d elements from the first file...." % num
elements = []
for n in range(num):
    number_strings = file.readline().split()
    elements.append(map(int, number_strings))
file.close()
print "Done."


#
# Read the second file.
#

#
# The space dimension and the simplex dimension.
#
file = open(sys.argv[2], "r")
number_strings = file.readline().split()
assert space_dimension == int(number_strings[0])
assert simplex_dimension == int(number_strings[1])

#
# Read the nodes.
#
num = int(file.readline())
print "Reading the %d node locations from the second file..." % num
for n in range(num):
    node_strings.append(file.readline())
print "Done."

#
# Read the elements.
#
num = int(file.readline())
print "Reading the %d elements from the second file...." % num
for n in range(num):
    number_strings = file.readline().split()
    elements.append(map(lambda x: x + num_nodes,
                          map(int, number_strings)))
file.close()
print "Done."


#
# Write the merged mesh.
#

print "Writing the merged mesh..."
file = open(sys.argv[3], "w")
file.write("%d %d\n" % (space_dimension, simplex_dimension))
file.write("%d\n" % len(node_strings))
for line in node_strings:
    file.write(line)

file.write("%d\n" % len(elements))
format = "%d" + " %d" * simplex_dimension + "\n"
for element in elements:
    file.write(format % tuple(element))
file.close()
print "Done."
