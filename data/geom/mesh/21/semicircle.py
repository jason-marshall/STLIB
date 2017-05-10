# semicircle.py
# Make a semicircle mesh.
# Usage:
# python semicircle.py x y radius num_points mesh

import sys, math

if len( sys.argv ) != 6:
    print "Usage:"
    print "python circle.py x y radius num_points mesh\n"
    raise "Bad command line arguments.  Exiting..."


# The center.
x = float(sys.argv[1])
y = float(sys.argv[2])

# The radius.
radius = float(sys.argv[3])

# The number of points.
numPoints = int(sys.argv[4])
if (numPoints < 3):
  raise("Bad number of points.")

# The mesh file.
file = open(sys.argv[5], "w")
file.write("2 1\n")
file.write("%d\n" % numPoints)
for n in range(numPoints):
  file.write("%g %g\n" % (x + radius * math.cos(n * math.pi / (numPoints-1)),
                          y + radius * math.sin(n * math.pi / (numPoints-1))))
file.write("%d\n" % numPoints)
for n in range(numPoints - 1):
  file.write("%d %d\n" % (n, n+1))
file.write("%d %d\n" % (numPoints - 1, 0))
file.close()
