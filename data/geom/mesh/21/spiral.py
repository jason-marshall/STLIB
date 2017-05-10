# spiral.py
# Make a spiral mesh.
# Usage:
# python spiral.py x y r0 r1 numPoints mesh

import sys, math

if len(sys.argv) != 7:
    print "Usage:"
    print "python spiral.py x y r0 r1 numPoints mesh\n"
    raise "Bad command line arguments.  Exiting..."


# The center.
x = float(sys.argv[1])
y = float(sys.argv[2])

# The starting radius.
r0 = float(sys.argv[3])
# The ending radius.
r1 = float(sys.argv[4])

# The number of points.
numPoints = int(sys.argv[5])
if (numPoints < 2):
  raise("Bad number of points.")

def r(t):
    return (1.0 - t) * r0 + t * r1

def f(t):
    return (x + r(t) * math.cos(2.0 * math.pi * t),
            y + r(t) * math.sin(2.0 * math.pi * t))

# The mesh file.
file = open(sys.argv[6], "w")
file.write("2 1\n")
file.write("%d\n" % numPoints)
for n in range(numPoints):
  file.write("%f %f\n" % f(n / (numPoints - 1.0)))
file.write("%d\n" % (numPoints-1))
for n in range(numPoints-1):
  file.write("%d %d\n" % (n, (n + 1) % numPoints))
file.close()
