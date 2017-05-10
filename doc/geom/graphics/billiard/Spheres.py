# Spheres.py
# Convert a triangle mesh in 3-D to Mathematica graphics of spheres.
# Usage:
# python Spheres.py <mesh.txt >spheres.dat

import sys, math, numpy

if len(sys.argv) != 1:
    print("Usage:")
    print("python Spheres.py <mesh.txt >spheres.dat\n")
    print("Convert a triangle mesh in 3-D to Mathematica graphics of spheres.")
    print("Bad command line arguments.  Exiting...")
    sys.exit(1)


# The space dimension and the simplex dimension.
numberStrings = sys.stdin.readline().split()
assert int(numberStrings[0]) == 3
assert int(numberStrings[1]) == 3

# Read the vertices.
numVertices = int(sys.stdin.readline())
vertices = []
for n in range(numVertices):
    numberStrings = sys.stdin.readline().split()
    vertices.append(tuple(map(float, numberStrings)))

# Read the tets.
numTets = int(sys.stdin.readline())
tets = []
for n in range(numTets):
    numberStrings = sys.stdin.readline().split()
    tets.append(tuple(map(int, numberStrings)))

# Write the Mathematica file.
lines = []
for indices in tets:
    t = [numpy.array(vertices[indices[i]]) for i in range(4)]
    c = sum(t) / 4
    r = 0
    for i in range(3):
        for j in range(i+1,4):
            m = 0.5*(t[i]+t[j])
            r = max(r, numpy.dot(c-m,c-m))
    r = math.sqrt(r)
    lines.append('Sphere[{%f,%f,%f},%f]' %
                 (c[0], c[1], c[2], r))
sys.stdout.write('{')
sys.stdout.write(',\n'.join(lines))
sys.stdout.write('}\n')
