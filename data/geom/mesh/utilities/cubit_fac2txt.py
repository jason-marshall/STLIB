# cubit_fac2txt.py
# Convert a simplicial mesh in Cubit facet format to an ascii file that 
# we can read.
# Usage:
# python cubit_fac2txt.py in out

import sys, string

if len(sys.argv) != 3:
    print "Usage:"
    print "python cubit_fac2txt.py in out\n"
    print "This program converts a Cubit facet mesh to a mesh in text format."
    raise "Bad command line arguments.  Exiting..."

# Open the input file.
in_file = open(sys.argv[1], "r")


#
# The number of vertices.
#
number_strings = string.split(in_file.readline())
if len(number_strings) != 1:
    raise "Error reading the number of vertices."
num_vertices = string.atoi(number_strings[0])
print "%d vertices." % num_vertices


#
# The vertices.
#
print "Reading the vertices..."
vertices = []
for n in range(num_vertices):
    number_strings = string.split(in_file.readline())
    if len(number_strings) != 4:
        raise "Error reading vertex number %d." % n
    vertices.append(string.join(number_strings[1:]))
print "Done."


#
# The number of faces.
#
number_strings = string.split(in_file.readline())
if len(number_strings) != 1:
    raise "Error reading the number of faces."
num_faces = string.atoi(number_strings[0])
print "%d faces." % num_faces


#
# The faces.
#
print "Reading the faces..."
faces = []
for n in range(num_faces):
    number_strings = string.split(in_file.readline())
    if len(number_strings) != 4:
        raise "Error reading face number %d." % n
    faces.append(string.join(number_strings[1:]))
print "Done."


#
# Close the input file.
#
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
    out_file.write("%s\n" % line)


#
# The faces.
#
out_file.write("%d\n" % num_faces)
for line in faces:
    out_file.write("%s\n" % line)


#
# Close the output file.
#
out_file.close()
