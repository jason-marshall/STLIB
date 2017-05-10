# Text2Mathematica.py
# Convert a triangle mesh in 3-D to Mathematica graphics.
# Usage:
# python Text2Mathematica.py <mesh.txt >mesh.dat

import sys

if len(sys.argv) != 1:
    print("Usage:")
    print("python Text2Mathematica.py <mesh.txt >mesh.dat\n")
    print("Convert a triangle mesh in 3-D to Mathematica graphics.")
    print("Bad command line arguments.  Exiting...")
    sys.exit(1)


# The space dimension and the simplex dimension.
numberStrings = sys.stdin.readline().split()
assert int(numberStrings[0]) == 3
assert int(numberStrings[1]) == 2

# Read the vertices.
numVertices = int(sys.stdin.readline())
vertices = []
for n in range(numVertices):
    numberStrings = sys.stdin.readline().split()
    vertices.append(tuple(map(float, numberStrings)))

# Read the triangles.
numTriangles = int(sys.stdin.readline())
triangles = []
for n in range(numTriangles):
    numberStrings = sys.stdin.readline().split()
    triangles.append(tuple(map(int, numberStrings)))

# Write the Mathematica file.
indices = []
for i in range(3):
    for j in range(3):
        indices.append((i,j))
lines = []
for t in triangles:
    lines.append('Polygon[{{%f,%f,%f},{%f,%f,%f},{%f,%f,%f}}]' %
                 tuple([vertices[t[i[0]]][i[1]] for i in indices]))
sys.stdout.write('{')
sys.stdout.write(',\n'.join(lines))
sys.stdout.write('}\n')
