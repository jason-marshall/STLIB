# The icosahedron has 12 vertices and 20 faces. Place a point at each vertex
# and at the center of each face.

import math
import random

#
# Define the vertices and indexed faces.
#
t = (1.0 + math.sqrt(5.0)) / 2.0
d = math.sqrt(1.0 + t * t)

vertices = [(t/d,  1/d,  0.),
            (-t/d,  1/d,  0.),
            (t/d, -1/d,  0.),
            (-t/d, -1/d,  0.),
            (1/d,  0.,  t/d),
            (1/d,  0., -t/d),
            (-1/d,  0.,  t/d),
            (-1/d,  0., -t/d),
            (0.,  t/d,  1/d),
            (0., -t/d,  1/d),
            (0.,  t/d, -1/d),
            (0., -t/d, -1/d)]

faces = [(0,  8,  4),
         (0,  5, 10),
         (2,  4,  9),
         (2, 11,  5),
         (1,  6,  8),
         (1, 10,  7),
         (3,  9,  6),
         (3,  7, 11),
         (0, 10,  8),
         (1,  8, 10),
         (2,  9, 11),
         (3, 11,  9),
         (4,  2,  0),
         (5,  0,  2),
         (6,  1,  3),
         (7,  3,  1),
         (8,  6,  4),
         (9,  4,  6),
         (10,  5,  7),
         (11,  7,  5)]

# Start with the vertices.
points = vertices[:]
# Add the midpoints of the faces.
for f in faces:
    # Average the three vertices of the face.
    x = (vertices[f[0]][0] + vertices[f[1]][0] + vertices[f[2]][0],
         vertices[f[0]][1] + vertices[f[1]][1] + vertices[f[2]][1],
         vertices[f[0]][2] + vertices[f[1]][2] + vertices[f[2]][2])
    # Normalize to place on the unit sphere.
    m = math.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
    x = (x[0]/m, x[1]/m, x[2]/m)
    points.append(x)
# Add the midpoints of the edges.
for f in faces:
    for i in range(3):
        j = (i + 1) % 3
        if (f[j] < f[i]):
            continue
        # Average the two vertices of the edge.
        x = (vertices[f[i]][0] + vertices[f[j]][0],
             vertices[f[i]][1] + vertices[f[j]][1],
             vertices[f[i]][2] + vertices[f[j]][2])
        # Normalize to place on the unit sphere.
        m = math.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
        x = (x[0]/m, x[1]/m, x[2]/m)
        points.append(x)
assert len(points) == 62

#
# Print the points on the sphere in a format that may be easily copied into
# a C++ file.
#
print('Points:')
for p in points:
    print('{%sf, %sf, %sf},' % p)

def normalized(x):
    im = 1. / math.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
    return (x[0]*im, x[1]*im, x[2]*im)

def averageNormalized(x, y):
    n = (x[0] + y[0], x[1] + y[1], x[2] + y[2])
    m = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
    return (n[0]/m, n[1]/m, n[2]/m)

def distance(x, y):
    return math.sqrt((x[0] - y[0]) * (x[0] - y[0]) +
                     (x[1] - y[1]) * (x[1] - y[1]) +
                     (x[2] - y[2]) * (x[2] - y[2]))

def randomUnitVector():
    x = (random.random(), random.random(), random.random())
    return normalized(x)

maxDistance = 0
for i in range(10000):
    x = randomUnitVector()
    d = 1e10
    for p in points:
        d = min(distance(x, p), d)
    maxDistance = max(maxDistance, d)
print(maxDistance)

# Lower bound.
a = vertices[0]
b = vertices[4]
c = points[12]
m = averageNormalized(a, b)
print(distance(c, m))

#print('\nThe farthest from a point on the sphere = %s.' % )
