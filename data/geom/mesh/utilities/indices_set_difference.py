# indices_set_difference.py
# The difference of two sets of indices.
#
# Usage:
# python indices_set_difference.py in1 in2 out

import sys, string

if len(sys.argv) != 4:
    print "Usage:"
    print "python indices_set_difference.py in1 in2 out\n"
    raise "Bad command line arguments.  Exiting..."

#
# Read the first file.
#
file = open(sys.argv[1], "r")
size = string.atoi(file.readline())
print "Reading the %d indices from the first file..." % size
a = []
for n in range(size):
    a.append(string.atoi(file.readline()))
file.close()
print "Done."

#
# Read the second file.
#
file = open(sys.argv[2], "r")
size = string.atoi(file.readline())
print "Reading the %d indices from the second file..." % size
b = {}
for n in range(size):
    b[ string.atoi(file.readline()) ] = None
file.close()
print "Done."

#
# Make the set difference.
#
c = []
for x in a:
    if not b.has_key(x):
        c.append(x)

#
# Write the set difference.
#
print "Writing the set difference..."
file = open(sys.argv[3], "w")
file.write("%d\n" % len(c))
for x in c:
    file.write("%d\n" % x)
file.close()
print "Done."
